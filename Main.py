from tqdm import tqdm
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 你的自定义模型
from model.PCT_Net import CNN_Transformer_AudioClassifier
from model.CNN import CNN_Baseline_AudioClassifier
from model.CNNBilSTM import CNN_BiLSTM_AudioClassifier
from model.Transformer import Transformer_AudioClassifier

from processed_librosa.data_loader import get_data_loaders

val_losses = []

# === 1. 标签字典映射 ===
train_df = pd.read_csv('divided_data/train.csv')
label_list = sorted(train_df['label'].unique())
label2idx = {label: idx for idx, label in enumerate(label_list)}
print("标签映射：", label2idx)

for fn in ["train", "val", "test"]:
    df = pd.read_csv(f'./divided_data/{fn}.csv')
    print(f"{fn} labels: {sorted(df['label'].unique())}")

# === 2. 训练参数 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("GPU名称:", torch.cuda.get_device_name(0))
else:
    print("当前为CPU运行")
num_epochs = 60
batch_size = 32
learning_rate = 1e-4

# === 3. Dataloader ===
train_loader, val_loader, test_loader = get_data_loaders(
    './divided_data/train.csv',
    './divided_data/val.csv',
    './divided_data/test.csv',
    batch_size=batch_size,
    label2idx=label2idx
)

# === 4. 模型选择（只保留一个激活）===
# model = CNN_Transformer_AudioClassifier(
#     sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
#     fmin=50, fmax=8000, classes_num=len(label2idx)
# ).to(device)

model = CNN_Baseline_AudioClassifier(
    sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
    fmin=50, fmax=8000, classes_num=len(label2idx)
).to(device)

# model = CNN_BiLSTM_AudioClassifier(
#     sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
#     fmin=50, fmax=8000, classes_num=len(label2idx)
# ).to(device)

# model = Transformer_AudioClassifier(
#     sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
#     fmin=50, fmax=8000, classes_num=len(label2idx)
# ).to(device)

# === 5. 损失函数与优化器 ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_accuracies = []
best_val_acc = 0.0
patience = 10
counter = 0
best_model_path = "best_model.pth"

# === 6. 训练主循环 ===
for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    running_loss = 0.0
    model.train()
    for batch_x, batch_y in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", ncols=80):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

    # === 7. 验证 ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            outputs = model(val_x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == val_y).sum().item()
            total += val_y.size(0)
    val_acc = correct / total
    val_accuracies.append(val_acc)
    # === 验证loss ===
    val_running_loss = 0.0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            outputs = model(val_x)
            loss = criterion(outputs, val_y)
            val_running_loss += loss.item() * val_x.size(0)
    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {epoch_val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"!!! New best model saved at epoch {epoch + 1} (val acc={val_acc:.4f}) !!!")
        counter = 0
    else:
        pass
        # counter += 1
        # if counter >= patience:
        #     print(f"Validation accuracy has not improved for {patience} epochs. Stopping early.")
        #     break

print("训练完成！")

# === 8. 测试集评估与混淆矩阵 ===
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for test_x, test_y in test_loader:
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        outputs = model(test_x)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_y.cpu().numpy())

test_acc = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"Test Accuracy: {test_acc:.4f}")

# === 9. 混淆矩阵与分类报告 ===
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)
label_names = [k for k, v in sorted(label2idx.items(), key=lambda x: x[1])]
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

# === 10. 绘图 ===
def smooth_curve(points, window_size=5):
    return pd.Series(points).rolling(window=window_size, min_periods=1).mean()

smooth_train_losses = smooth_curve(train_losses, window_size=7)
smooth_val_losses = smooth_curve(val_losses, window_size=7)

plt.figure(figsize=(8,6))
plt.plot(train_losses, color='red', label='train loss')
plt.plot(val_losses, color='green', label='val loss')
plt.plot(smooth_train_losses, color='black', linestyle='--', label='smooth train loss')
plt.plot(smooth_val_losses, color='orange', linestyle='--', label='smooth val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train & Validation Loss Curve')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy Curve')
plt.grid(True)
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix (Test Set)")
plt.show()