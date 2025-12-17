import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Baseline_AudioClassifier(nn.Module):
    def __init__(self, sample_rate=16000, window_size=512, hop_size=128, mel_bins=64, fmin=50, fmax=8000, classes_num=3):
        super(CNN_Baseline_AudioClassifier, self).__init__()
        # 直接处理梅尔谱输入，假定输入shape为[batch, 1, mel_bins, time]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d((2,2))
        self.dropout = nn.Dropout(0.3)

        # 最后做全局池化，自动适配长度
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, classes_num)

    def forward(self, x):
        # x: [batch, 1, mel_bins, time]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = self.global_pool(x)  # [batch, 64, 1, 1]
        x = x.view(x.size(0), -1) # flatten [batch, 64]
        out = self.fc(x)          # [batch, classes_num]
        return out