import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_BiLSTM_AudioClassifier(nn.Module):
    def __init__(self, sample_rate=16000, window_size=512, hop_size=128, mel_bins=64, fmin=50, fmax=8000,
                 classes_num=3):
        super(CNN_BiLSTM_AudioClassifier, self).__init__()
        # === CNN部分，提取局部特征 ===
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.3)

        # === BiLSTM部分，做序列建模 ===
        # 假设经过两次2x2池化后 mel_bins/4, time/4
        self.mel_bins = mel_bins // 4
        self.lstm_input_size = self.mel_bins * 32  # channel*mel_bins
        self.hidden_size = 64
        self.lstm_layers = 2
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.hidden_size,
                              num_layers=self.lstm_layers, batch_first=True, bidirectional=True)

        # === 分类器 ===
        self.fc = nn.Linear(self.hidden_size * 2, classes_num)  # 双向LSTM输出

    def forward(self, x):
        # x: [batch, 1, mel_bins, time]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> [batch, 16, mel_bins/2, time/2]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> [batch, 32, mel_bins/4, time/4]
        x = self.dropout(x)

        # LSTM输入为 [batch, seq_len, feat_dim]
        # 先交换轴，[batch, channel, mel, time] -> [batch, time, channel, mel]
        x = x.permute(0, 3, 1, 2)  # -> [batch, time/4, 32, mel_bins/4]
        B, T, C, M = x.shape
        x = x.contiguous().view(B, T, C * M)  # [batch, time, 32*mel_bins/4]

        lstm_out, _ = self.bilstm(x)  # [batch, time, hidden*2]
        out = lstm_out[:, -1, :]  # 取最后时刻输出 [batch, hidden*2]
        out = self.fc(out)  # [batch, classes_num]
        return out