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

# ====== SNR 混噪评测工具（按需加载版，节省内存） ======
import torchaudio as ta
from pathlib import Path
import random, math
import numpy as np

# 为了可复现
random.seed(0); np.random.seed(0); torch.manual_seed(0)

def _mix_to_snr(clean_1d: torch.Tensor, noise_1d: torch.Tensor, snr_db: float) -> torch.Tensor:
    """输入 1D clean/noise -> 1D 混合，使 10log10(Ps/Pn)=snr_db"""
    if clean_1d.dim() != 1: clean_1d = clean_1d.view(-1)
    if noise_1d.dim() != 1: noise_1d = noise_1d.view(-1)
    T = clean_1d.shape[0]
    if noise_1d.shape[0] < T:
        rep = math.ceil(T / noise_1d.shape[0])
        noise_1d = noise_1d.repeat(rep)[:T]
    else:
        s = random.randint(0, noise_1d.shape[0] - T)
        noise_1d = noise_1d[s:s+T]

    Ps = (clean_1d**2).mean().item()
    Pn = (noise_1d**2).mean().item() + 1e-12
    a = math.sqrt(Ps / (Pn * (10.0**(snr_db/10.0))))
    mix = clean_1d + a * noise_1d
    return torch.clamp(mix, -1., 1.)

def _scan_noise_paths(root: str) -> dict:
    """
    返回 {scene_name: [Path, Path, ...]}；不把音频读进内存，只存路径，适合你每类 ~500 条的情况
    - root 下面的每个子文件夹视为一个噪声场景
    """
    root_p = Path(root)
    dirs = [p for p in root_p.iterdir() if p.is_dir()] or [root_p]
    bank = {}
    exts = {".wav", ".flac", ".mp3", ".ogg"}
    for d in dirs:
        paths = [p for p in d.rglob("*") if p.suffix.lower() in exts]
        if paths:
            bank[d.name] = paths
    return bank

@torch.no_grad()
def _load_random_noise_segment(noise_path: Path, target_len: int, sample_rate: int) -> torch.Tensor:
    """从一个噪声文件随机取出 target_len 长度的单声道段落，返回 1D Tensor"""
    w, sr = ta.load(str(noise_path))      # (C, N)
    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        w = ta.functional.resample(w, sr, sample_rate)
    w = w.squeeze(0).contiguous()
    if w.shape[0] < target_len:
        rep = math.ceil(target_len / w.shape[0])
        w = w.repeat(rep)[:target_len]
    else:
        s = random.randint(0, w.shape[0] - target_len)
        w = w[s:s+target_len]
    return w

@torch.inference_mode()
def eval_with_noise(model, loader, device, noise_paths_bank: dict, snr_db: float, sample_rate: int = 16000):
    """
    对每个场景在指定 SNR 下做一次完整测试，返回 {scene: acc}
    - loader 必须输出波形 (B,1,T) 或 (B,T)
    """
    model.eval()
    results = {}
    for scene, path_list in noise_paths_bank.items():
        correct, total = 0, 0
        for xb, yb in loader:
            if xb.dim() == 2:    # (B,T) -> (B,1,T)
                xb = xb.unsqueeze(1)
            B, _, T = xb.shape
            mix_batch = []
            for i in range(B):
                clean_i = xb[i, 0].cpu()                         # 1D
                noise_p = random.choice(path_list)
                noise_i = _load_random_noise_segment(noise_p, T, sample_rate)
                mix_i = _mix_to_snr(clean_i, noise_i, snr_db)    # 1D
                mix_batch.append(mix_i.unsqueeze(0))             # (1,T)
            mix_batch = torch.stack(mix_batch, dim=0).to(device)  # (B,1,T)
            yb = yb.to(device)
            pred = model(mix_batch).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
        results[scene] = correct / max(1, total)
    return results

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
model = CNN_Transformer_AudioClassifier(
    sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
    fmin=50, fmax=8000, classes_num=len(label2idx)
).to(device)

# model = CNN_Baseline_AudioClassifier(
#     sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
#     fmin=50, fmax=8000, classes_num=len(label2idx)
# ).to(device)

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

# ====== SNR 鲁棒性评测（同一张图 4 条曲线：3 场景 + clean） ======
NOISE_ROOT = "./processed_librosa/Noise111"   # 你的目录：里面有 noise1|noise2|noise3
SNR_LIST   = [20, 10, 5, 0, -5]               # 横轴；从高到低表现成“越来越吵”

noise_paths_bank = _scan_noise_paths(NOISE_ROOT)
if not noise_paths_bank:
    print(f"[Warn] 在 {NOISE_ROOT} 下没找到子目录/噪声音频，跳过鲁棒性评测。")
else:
    print(f"[Info] 噪声场景：{list(noise_paths_bank.keys())}")

    # 1) clean 基线（不随 SNR 变化，画成一条虚线）
    clean_acc = [test_acc for _ in SNR_LIST]

    # 2) 每个噪声场景：在不同 SNR 的准确率
    scene_acc = {scene: [] for scene in noise_paths_bank.keys()}
    for snr in SNR_LIST:
        res = eval_with_noise(model, test_loader, device, noise_paths_bank, snr, sample_rate=16000)
        for scene, acc in res.items():
            scene_acc[scene].append(acc)
        print(f"SNR={snr} dB -> {res}")

    # 3) 画图
    plt.figure(figsize=(8,6))
    # 噪声场景曲线
    for scene, accs in scene_acc.items():
        plt.plot(SNR_LIST, accs, marker='o', linewidth=2, label=scene)
    # clean 基线
    plt.plot(SNR_LIST, clean_acc, linestyle='--', linewidth=2, label='clean (no noise)')
    plt.gca().invert_xaxis()  # 让 SNR 从 20 → -5，视觉上“越往右越吵”
    plt.xlabel("SNR (dB)  →  lower means noisier")
    plt.ylabel("Accuracy")
    plt.title("Noise robustness across real-world scenes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) 保存 csv（便于制表/画论文图）
    rows = []
    for scene, accs in scene_acc.items():
        for snr, acc in zip(SNR_LIST, accs):
            rows.append({"scene": scene, "snr_db": snr, "accuracy": acc})
    pd.DataFrame(rows).to_csv("snr_robustness_results.csv", index=False)
    print("[Saved] snr_robustness_results.csv")

    # 5) 可选：给每个场景算一个“鲁棒性面积分”指标（越大越好）
    def auc_on_snr(xs, ys):
        # 简单梯形积分；xs 单位 dB；返回 0~1 的“平均准确率”
        if len(xs) < 2: return float(ys[0]) if ys else 0.0
        # 归一化横轴宽度
        width = abs(xs[0] - xs[-1])
        area = 0.0
        for i in range(len(xs)-1):
            dx = abs(xs[i] - xs[i+1]) / width
            area += 0.5 * (ys[i] + ys[i+1]) * dx
        return float(area)

    print("\n[Noise-AUC] 场景鲁棒性面积分（0~1，越大越好）：")
    for scene, accs in scene_acc.items():
        print(f"  {scene:12s}: AUC={auc_on_snr(SNR_LIST, accs):.3f}")

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