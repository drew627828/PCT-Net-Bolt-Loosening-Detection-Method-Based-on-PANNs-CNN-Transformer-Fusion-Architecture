import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== PANNs经典组件 =====
class Spectrogram(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        window_tensor = torch.hann_window(win_length)
        self.register_buffer('window_tensor', window_tensor)
        if freeze_parameters:
            self.window_tensor.requires_grad = False

    def forward(self, x):
        # x: (batch, data_len)
        x = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window_tensor, center=self.center, pad_mode=self.pad_mode,
            return_complex=True
        )
        spectro = torch.abs(x)  # (batch, freq_bins, time_steps)
        # (batch, 1, time_steps, freq_bins)
        return spectro.unsqueeze(1).transpose(2, 3)


class LogmelFilterBank(nn.Module):
    def __init__(self, sr, n_fft, n_mels, fmin, fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True):
        super(LogmelFilterBank, self).__init__()
        import librosa
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, x):
        # x: (batch, 1, time_steps, freq_bins)
        x = x.squeeze(1)  # (batch, time_steps, freq_bins)
        mel_spec = torch.matmul(x, self.mel_basis.T)
        logmel = 10.0 * torch.log10(torch.clamp(mel_spec, min=self.amin))
        return logmel.unsqueeze(1)  # (batch, 1, time_steps, mel_bins)


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2):
        super(SpecAugmentation, self).__init__()
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

    def forward(self, x):
        # x: (batch, 1, time_steps, mel_bins)
        batch_size, ch, time_steps, mel_bins = x.shape
        x = x.clone()
        # Time masking
        for i in range(self.time_stripes_num):
            t = torch.randint(0, time_steps - self.time_drop_width, (1,))
            x[:, :, t:t+self.time_drop_width, :] = 0
        # Frequency masking
        for i in range(self.freq_stripes_num):
            f = torch.randint(0, mel_bins - self.freq_drop_width, (1,))
            x[:, :, :, f:f+self.freq_drop_width] = 0
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, pool_size=(2,2), pool_type='avg'):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError('Unsupported pool type: {}'.format(pool_type))
        return x


def do_mixup(x, lmbd):
    ''' x: shape (batch, ...) '''
    # 假设已经随机打乱过索引
    batch_size = x.shape[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lmbd * x + (1 - lmbd) * x[index]
    return mixed_x


# =========== 主模型 ==============
class CNN_Transformer_AudioClassifier(nn.Module):
    def __init__(self,
                 sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 classes_num,
                 cnn_channels=[64, 128, 256, 512],
                 transformer_layers=2, transformer_heads=8,
                 embedding_dim=512, dropout=0.5):
        super().__init__()
        # 1. 频谱转换与增强（PANNs）
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
                                                 fmax=fmax)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8,
                                               freq_stripes_num=2)
        # 2. 多层CNN特征提取（PANNs ConvBlock精华）
        self.cnn_blocks = nn.Sequential(
            ConvBlock(1, cnn_channels[0]),
            ConvBlock(cnn_channels[0], cnn_channels[1]),
            ConvBlock(cnn_channels[1], cnn_channels[2]),
            ConvBlock(cnn_channels[2], cnn_channels[3]),
        )
        self.bn = nn.BatchNorm2d(cnn_channels[3])
        # 3. Transformer Encoder（AST思路，进行全局建模）
        self.embedding_dim = embedding_dim
        # 如果cnn输出channels ≠ embedding_dim，需线性变换
        self.fc_cnn2emb = nn.Linear(cnn_channels[3], embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=transformer_heads, dim_feedforward=embedding_dim, dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        # 4. 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embedding_dim, classes_num)

    def forward(self, x, mixup_lambda=None):
        # x: (batch, 1, samples) or (batch, samples)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # 去掉 channel 维
        # ----- 1. 频谱前端 -----
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        if self.training:
            x = self.spec_augmenter(x)
        # [B, 1, T, F]
        # ----- 2. CNN特征提取 -----
        x = self.cnn_blocks(x)
        x = self.bn(x)
        # [B, C, T', F']
        # 做平均池化以得到每个时间步的全频特征
        x = torch.mean(x, dim=3)  # [B, C, T']
        x = x.transpose(1, 2)  # [B, T', C] → [T', B, C]
        # ----- 3. FC变换 + Transformer编码 -----
        x = self.fc_cnn2emb(x)  # [T', B, embedding_dim]
        x = x.transpose(0, 1)  # [B, T', embedding_dim] → [T', B, embedding_dim]
        x = self.transformer(x)  # [T', B, embedding_dim]
        # ----- 4. 池化聚合与分类 -----
        x = x.mean(dim=0)  # 全局平均池化: [B, embedding_dim]
        x = self.dropout(x)
        out = self.fc_out(x)
        return out


# ====== 示例用法 ======
if __name__ == '__main__':
    model = CNN_Transformer_AudioClassifier(
        sample_rate=32000, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=3
    )
    x = torch.randn(4, 16000 * 5)  # 4条5秒音频
    y = model(x)
    print(y.shape)  # torch.Size([4, 3])