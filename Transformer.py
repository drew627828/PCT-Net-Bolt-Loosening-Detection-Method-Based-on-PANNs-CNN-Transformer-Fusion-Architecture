import torch
import torch.nn as nn
import torch.nn.functional as F

class Spectrogram(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window='hann', center=True, pad_mode='reflect'):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_tensor = torch.hann_window(win_length)
        self.center = center
        self.pad_mode = pad_mode

    def forward(self, x):
        # x: (batch, data_len)
        x = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window_tensor.to(x.device), center=self.center, pad_mode=self.pad_mode,
            return_complex=True
        )
        spectro = torch.abs(x)  # (batch, freq_bins, time_steps)
        return spectro.unsqueeze(1).transpose(2, 3)

class LogmelFilterBank(nn.Module):
    def __init__(self, sr, n_fft, n_mels, fmin, fmax, amin=1e-10):
        super().__init__()
        import librosa
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        self.amin = amin

    def forward(self, x):
        x = x.squeeze(1)  # (batch, time_steps, freq_bins)
        mel_spec = torch.matmul(x, self.mel_basis.T)
        logmel = 10.0 * torch.log10(torch.clamp(mel_spec, min=self.amin))
        return logmel.unsqueeze(1)  # (batch, 1, time_steps, mel_bins)

class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2):
        super().__init__()
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

    def forward(self, x):
        batch_size, ch, time_steps, mel_bins = x.shape
        x = x.clone()
        # Time masking
        for i in range(self.time_stripes_num):
            t = torch.randint(0, max(1, time_steps - self.time_drop_width), (1,))
            x[:, :, t:t+self.time_drop_width, :] = 0
        # Frequency masking
        for i in range(self.freq_stripes_num):
            f = torch.randint(0, max(1, mel_bins - self.freq_drop_width), (1,))
            x[:, :, :, f:f+self.freq_drop_width] = 0
        return x

class Transformer_AudioClassifier(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num,
                 transformer_layers=4, transformer_heads=8, embedding_dim=128, dropout=0.3):
        super().__init__()
        # 前端特征提取
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, win_length=window_size)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax)
        self.spec_augmenter = SpecAugmentation()
        # 直接做线性embedding
        self.embedding_dim = embedding_dim
        self.input_fc = nn.Linear(mel_bins, embedding_dim)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=transformer_heads, dim_feedforward=embedding_dim*2, dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embedding_dim, classes_num)

    def forward(self, x):
        # x: (batch, 1, num_samples)
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        if self.training:
            x = self.spec_augmenter(x)
        # x: [B, 1, T, F] → [B, T, F]
        x = x.squeeze(1)
        x = self.input_fc(x)  # [B, T, embedding_dim]
        x = x.transpose(0, 1)  # [T, B, embedding_dim]
        x = self.transformer(x)  # [T, B, embedding_dim]
        x = x.mean(dim=0)  # [B, embedding_dim]
        x = self.dropout(x)
        out = self.fc_out(x)
        return out

# ==== 用法举例 ====
if __name__ == '__main__':
    model = Transformer_AudioClassifier(
        sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
        fmin=50, fmax=8000, classes_num=2
    )
    x = torch.randn(4, 1, 16000*5)  # 4条5秒音频
    y = model(x)
    print(y.shape)  # [4, 2]