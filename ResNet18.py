import torch
import torch.nn as nn
import torchaudio as ta
from torchvision.models import resnet18

class LogMelSpec(nn.Module):
    """把波形 (B,T) 或 (B,1,T) -> Log-Mel (B, n_mels, T') 并做标准化"""
    def __init__(
        self,
        sr=16000, n_fft=512, hop_length=128, win_length=512,
        n_mels=64, fmin=0, fmax=None,
        mean=None, std=None
    ):
        super().__init__()
        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, center=True, power=2.0
        )
        self.to_db = ta.transforms.AmplitudeToDB(stype="power")
        # 数据集全局均值/方差（可之后再回填）
        if mean is None:
            self.register_buffer("global_mean", torch.tensor(0.0), persistent=False)
            self.register_buffer("global_std",  torch.tensor(1.0), persistent=False)
            self.use_global = False
        else:
            self.register_buffer("global_mean", torch.tensor(float(mean)))
            self.register_buffer("global_std",  torch.tensor(float(std) if std is not None else 1.0))
            self.use_global = True
        # 备用：按样本进行实例归一化（当没有全局 mean/std 时用）
        self.instancenorm = nn.InstanceNorm2d(1, affine=False)

    def forward(self, x):  # x: (B,T) 或 (B,1,T)
        if x.dim() == 3:
            x = x.squeeze(1)                       # -> (B,T)
        S = self.melspec(x)                        # (B, n_mels, T')
        S = self.to_db(S)                          # log 能量 (dB)
        S = S.unsqueeze(1)                         # -> (B,1,n_mels,T') 给 CNN

        if self.use_global:
            S = (S - self.global_mean) / (self.global_std + 1e-6)
        else:
            S = self.instancenorm(S)
        return S  # (B,1,n_mels,T')


class ResNet18Audio(nn.Module):
    """
    输入：波形 (B,T) / (B,1,T)
    输出：logits (B,num_classes)
    """
    def __init__(
        self, num_classes,
        sr=16000, n_fft=512, hop_length=128, win_length=512,
        n_mels=64, fmin=0, fmax=None,
        specaug=True,  # 只在训练时启用
        mean=None, std=None
    ):
        super().__init__()
        self.frontend = LogMelSpec(sr, n_fft, hop_length, win_length, n_mels, fmin, fmax, mean, std)
        # 轻量的 SpecAugment（在谱图上做）
        self.specaug = specaug
        self.freq_mask = ta.transforms.FrequencyMasking(freq_mask_param=8)
        self.time_mask = ta.transforms.TimeMasking(time_mask_param=32)

        # ResNet18 主干（改成 1 通道输入 + 输出类别数）
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):  # x: (B,T) or (B,1,T)
        x = self.frontend(x)            # (B,1,n_mels,T')
        if self.training and self.specaug:
            # SpecAug 只在训练时做
            # 逐样本随机遮挡；torchaudio 期望 (B,1,F,T)
            x = self.freq_mask(x)
            x = self.time_mask(x)
        logits = self.backbone(x)       # (B,num_classes)
        return logits