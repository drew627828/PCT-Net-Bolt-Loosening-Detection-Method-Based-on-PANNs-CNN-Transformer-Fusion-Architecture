from tqdm import tqdm
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import time
import contextlib
torch.backends.cudnn.benchmark = True  # 提升GPU计时稳定性
try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

# 你的自定义模型
from model.PCT_Net import CNN_Transformer_AudioClassifier
from model.CNN import CNN_Baseline_AudioClassifier
from model.CNNBilSTM import CNN_BiLSTM_AudioClassifier
from model.Transformer import Transformer_AudioClassifier
from model.ResNET18 import ResNet18Audio

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
num_epochs = 20
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
MODEL_NAME = "ResNET18"
# model = CNN_Transformer_AudioClassifier(
#     sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
#     fmin=50, fmax=8000, classes_num=len(label2idx)
# ).to(device)

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

model = ResNet18Audio(
    num_classes=len(label2idx),
    sr=16000, n_fft=512, hop_length=128, win_length=512,
    n_mels=64, fmin=50, fmax=8000,
    specaug=True,          # 训练时自动开启
    mean=None, std=None    # 没有全局统计就留 None，会用 InstanceNorm
).to(device)

# ========== Efficiency helpers ==========
def count_params(model: torch.nn.Module):
    """可训练参数量（返回整数 & 百万参数数值）"""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n, n / 1e6

@torch.no_grad()
def measure_flops(model, example_input):
    """用 THOP 估 FLOPs（单样本）。失败则返回 None。"""
    if thop_profile is None:
        return None
    model.eval()
    try:
        flops, _ = thop_profile(model, inputs=(example_input,), verbose=False)
        return flops  # 标量（单样本一次前向）
    except Exception:
        return None

@torch.no_grad()
def measure_latency(model, example_input, n_warmup=20, n_runs=100, use_cuda_sync=True):
    """
    单样本推理时延（ms）；example_input 形状要与真实一致，建议 batch=1。
    """
    model.eval()
    device = next(model.parameters()).device
    x = example_input.clone().to(device)

    # 预热
    for _ in range(n_warmup):
        _ = model(x)
    if device.type == "cuda" and use_cuda_sync:
        torch.cuda.synchronize()

    # 计时
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(x)
    if device.type == "cuda" and use_cuda_sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / n_runs * 1000.0
    return avg_ms

def epoch_to_reach(acc_list, target_ratio=0.95):
    """
    第一次达到“最终最优精度 * target_ratio”的 epoch（从 1 开始计数）。
    若未达到，返回 len(acc_list)。
    """
    if not acc_list:
        return 0
    best = max(acc_list)
    thr = best * target_ratio
    for i, a in enumerate(acc_list, start=1):
        if a >= thr:
            return i
    return len(acc_list)

# ====== 基准输入（从 val_loader 取 1 条）并测 Params / FLOPs / Latency ======
with torch.no_grad():
    _bx, _by = next(iter(val_loader))   # _bx: (B, 1, T) 或 (B,T)
    _bx = _bx[:1].to(device)            # 单样本
# 参数量
n_params, n_params_m = count_params(model)
# FLOPs（可能因不支持而返回 None）
flops_single = measure_flops(model, _bx)  # 单样本单前向 FLOPs
# 推理时延（ms/样本）
latency_ms = measure_latency(model, _bx, n_warmup=20, n_runs=50)
# 吞吐/FPS（样本/秒）
fps = 1000.0 / latency_ms

print("\n[Model Efficiency @ single sample]")
print(f"Params      : {n_params_m:.3f} M ({n_params:,})")
if flops_single is not None:
    print(f"FLOPs       : {flops_single/1e9:.3f} GFLOPs per sample")
else:
    print("FLOPs       : (THOP 不支持该模型，已跳过)")
print(f"Latency     : {latency_ms:.2f} ms / sample")
print(f"Throughput  : {fps:.1f} samples/sec\n")

# === 5. 损失函数与优化器 ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_accuracies = []
best_val_acc = 0.0
patience = 10
counter = 0
best_model_path = f"best_model_{MODEL_NAME}.pth"

# === 6. 训练主循环 ===
import time
epoch_times = []
for epoch in range(num_epochs):
    t_epoch0 = time.perf_counter()
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
    # ==== 本 epoch 结束计时（新增） ====
    t_epoch1 = time.perf_counter()
    epoch_time = t_epoch1 - t_epoch0
    epoch_times.append(epoch_time)
    print(f"[Epoch {epoch+1}] time: {epoch_time:.2f} s")

# ==== 训练循环结束后做一个汇总（覆盖你现有的这块） ====
mean_epoch_s = float(np.mean(epoch_times)) if epoch_times else 0.0   # ← 改：无条件求值
print("\n[Training Efficiency]")
print(f"Avg time / epoch : {mean_epoch_s:.2f} s")

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

# ==== 新增：先计算 t95_epoch，再汇总 ====
t95_epoch = epoch_to_reach(val_accuracies, target_ratio=0.95)   # ← 新增

summary = {
    "model": "ResNET18",
    "params_M": round(n_params_m, 3),
    "FLOPs_G": round((flops_single or 0)/1e9, 3),
    "latency_ms": round(latency_ms, 2),
    "fps": round(fps, 1),
    "avg_epoch_s": round(mean_epoch_s, 1),
    "best_val_acc": round(max(val_accuracies), 4) if val_accuracies else None,
    "epoch_at_95%best": t95_epoch,            # ← 现在已定义
    "test_acc": round(float(test_acc), 4)
}
print("\n[Summary]")
for k, v in summary.items():
    print(f"{k:>16s}: {v}")

# 保存 CSV（连续跑多个模型时会很方便）
import csv, os
csv_path = "efficiency_summary.csv"
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary.keys()))
    if write_header:
        w.writeheader()
    w.writerow(summary)
print(f"[Saved] {csv_path}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("efficiency_summary.csv")

# 确保这些列为数值类型（防止被当作字符串）
for col in ["test_acc", "FLOPs_G", "params_M", "latency_ms", "fps", "avg_epoch_s", "best_val_acc"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# === 方案A：按“每个模型”保留 test_acc 最高的一条（用于权衡图） ===
idx = df.groupby("model")["test_acc"].idxmax()           # 找到每个模型 test_acc 最大的行索引
df_best = df.loc[idx].reset_index(drop=True)             # 只保留这些“最优”行

# === 方案B：全局只要测试精度最高的那一条（如果你想单独打印/标注） ===
row_best = df.loc[df["test_acc"].idxmax()]               # 单行 Series
print("\n[Overall Best by Test Accuracy]")
print(row_best.to_dict())

# ↓↓↓ 下面画图时，把原来的 df 改为 df_best ↓↓↓
def pareto_frontier(df, xcol, ycol, minimize_x=True, maximize_y=True):
    d = df[[xcol, ycol, "model"]].dropna().copy()
    d = d.sort_values(by=xcol, ascending=minimize_x).reset_index(drop=True)
    frontier_idx, best_y = [], (-np.inf if maximize_y else np.inf)
    for i, row in d.iterrows():
        y = row[ycol]
        if (maximize_y and y > best_y) or ((not maximize_y) and y < best_y):
            frontier_idx.append(i); best_y = y
    return d.loc[frontier_idx]

def draw_tradeoff(df_plot, xcol, xlabel, logx=False, fname=None):
    plt.figure(figsize=(6.8, 4.8))
    plt.scatter(df_plot[xcol], df_plot["test_acc"], s=60, alpha=0.85)
    for _, r in df_plot.iterrows():
        plt.annotate(r["model"], (r[xcol], r["test_acc"]), xytext=(5,5), textcoords="offset points", fontsize=9)
    pf = pareto_frontier(df_plot, xcol, "test_acc", minimize_x=True, maximize_y=True)
    if len(pf) >= 2:
        plt.plot(pf[xcol], pf["test_acc"], linewidth=2)
    if logx: plt.xscale("log")
    plt.xlabel(xlabel); plt.ylabel("Test Accuracy"); plt.title(f"Accuracy vs. {xlabel}")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    if fname: plt.savefig(fname, dpi=300)
    plt.show()

if (df_best["FLOPs_G"] > 0).any():
    draw_tradeoff(df_best[df_best["FLOPs_G"] > 0], "FLOPs_G", "GFLOPs / sample (log)", logx=True, fname="tradeoff_flops.png")

if (df_best["params_M"] > 0).any():
    draw_tradeoff(df_best[df_best["params_M"] > 0], "params_M", "Params (Million, log)", logx=True, fname="tradeoff_params.png")

if (df_best["latency_ms"] > 0).any():
    draw_tradeoff(df_best[df_best["latency_ms"] > 0], "latency_ms", "Latency (ms / sample)", logx=False, fname="tradeoff_latency.png")

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