from tqdm import tqdm
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import torchaudio as ta
import warnings

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

model = ResNet18Audio(
    num_classes=len(label2idx),
    sr=16000, n_fft=512, hop_length=128, win_length=512, n_mels=64,
    target_size=224, pretrained=False, specaug=True
).to(device)


# ========== Sanity Check：训练前快速自检（5~10 秒） ==========
RUN_SANITY_CHECK = True
if RUN_SANITY_CHECK:
    import warnings
    warnings.simplefilter("default")

    # 1) 特征提取器：Log-Mel(dB)+Δ/ΔΔ -> 1D 向量
    sr = 16000
    melspec = ta.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=512, hop_length=128, win_length=512,
        n_mels=64, center=True, power=2.0
    )
    to_db = ta.transforms.AmplitudeToDB(stype='power')  # 产生 -inf 时我们手动处理

    @torch.no_grad()
    def _extract_feat(wave_1d: torch.Tensor) -> np.ndarray:
        # 输入 (T,) 或 (1,T)，输出 1D，已做数值清理 + L2 归一化
        if wave_1d.dim() == 2:
            wave_1d = wave_1d.squeeze(0)

        # Mel power -> dB
        S = melspec(wave_1d)                       # (F, T'), power>=0
        S_db = to_db(S)                            # 可能含 -inf
        S_db = torch.nan_to_num(S_db, nan=-80.0, posinf=80.0, neginf=-80.0)
        S_db = S_db.clamp(min=-80.0, max=80.0)

        # Δ / ΔΔ 在 dB 上计算
        delta  = ta.functional.compute_deltas(S_db)
        delta2 = ta.functional.compute_deltas(delta)

        # 拼接并展平 -> 1D
        F = torch.cat([S_db, delta, delta2], dim=0)    # (3F, T')
        v = F.reshape(-1)                               # (3F*T',)

        # 数值保险 & 归一化
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        denom = v.norm(p=2)
        if torch.isfinite(denom) and denom > 0:
            v = v / denom
        return v.cpu().numpy()

    # 2) 抽样少量测试样本做全流程验证
    feats, labs = [], []
    max_samples = 64
    collected = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            for xi, yi in zip(xb, yb):
                feats.append(_extract_feat(xi.cpu()))
                labs.append(int(yi))
                collected += 1
                if collected >= max_samples:
                    break
            if collected >= max_samples:
                break

    feats = np.asarray(feats)   # (N, D)；若不是二维，下面会强制展平
    labs  = np.asarray(labs)

    # 3) 维度/数值清理（统一用 feats/labs，不再混用 arr/features/labels_idx）
    if feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    row_norm = np.linalg.norm(feats, axis=1)
    valid = np.isfinite(row_norm) & (row_norm > 0)
    if valid.sum() < feats.shape[0]:
        print(f"[Warn] 丢弃 {feats.shape[0] - valid.sum()} 个异常样本（NaN/Inf/全零）")
    feats = feats[valid]
    labs  = labs[valid]

    inv_label_map = {v: k for k, v in label2idx.items()}
    labels_txt = np.array([inv_label_map[int(i)] for i in labs])
    classes = sorted(set(labels_txt))
    assert len(classes) >= 2, "Sanity check 样本太少，至少覆盖两个类别"

    # 4) 质心、类内/类间距离（稳健版）
    centroids_list, intra = [], {}
    for c in classes:
        Xc = feats[labels_txt == c]  # (Nc, D)
        if Xc.size == 0:
            continue
        # 数值清理
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        mu = Xc.mean(axis=0, keepdims=True)  # (1, D)
        centroids_list.append(mu.squeeze(0))
        d = np.linalg.norm(Xc - mu, axis=1)
        d = d[np.isfinite(d)]
        intra[c] = float(d.mean()) if d.size > 0 else 0.0

    # 质心矩阵
    centroids = np.stack(centroids_list, axis=0) if len(centroids_list) > 0 else np.zeros((0, feats.shape[1]))
    centroids = np.asarray(centroids, dtype=np.float64)
    centroids = np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)

    # —— 关键：自己算欧氏距离（避免 sklearn 内部 NaN→int 的坑）——
    if centroids.shape[0] >= 2 and centroids.shape[1] > 0:
        diff = centroids[:, None, :] - centroids[None, :, :]  # (K, K, D)
        Dmat = np.sqrt(np.sum(diff * diff, axis=2))  # (K, K)
    else:
        # 类别不足或维度异常时的兜底
        Dmat = np.zeros((centroids.shape[0], centroids.shape[0]), dtype=np.float64)

    # 汇总指标
    mean_intra = float(np.mean(list(intra.values()))) if intra else 0.0
    tri = np.triu_indices(Dmat.shape[0], k=1)
    mean_inter = float(Dmat[tri].mean()) if Dmat.size and tri[0].size > 0 else 0.0
    ratio = mean_inter / (mean_intra + 1e-8) if mean_intra > 0 else 0.0

    print("\n[Sanity Check] 预检查：")
    print(f" - 抽样 N={feats.shape[0]}, D={feats.shape[1]}, 类别 K={len(classes)}")
    print(f" - mean intra={mean_intra:.4f}, mean inter={mean_inter:.4f}, ratio={ratio:.3f}")

    # 可视化热力图（用我们自己算的 Dmat）
    if Dmat.shape[0] >= 2:
        plt.figure(figsize=(6, 5))
        plt.imshow(Dmat, cmap='Blues')
        plt.colorbar(label='Euclidean distance')
        plt.xticks(range(len(classes)), classes, rotation=60, ha='right')
        plt.yticks(range(len(classes)), classes)
        plt.title("Centroid-to-Centroid Distance (Euclidean)")
        plt.tight_layout()
        plt.show()
# ========== /Sanity Check ==========

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
        if batch_x.dim() == 3 and batch_x.size(1) == 1:
            batch_x = batch_x.squeeze(1)  # (B,1,T) -> (B,T)
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
            if batch_x.dim() == 3 and batch_x.size(1) == 1:
                batch_x = batch_x.squeeze(1)  # (B,1,T) -> (B,T)
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
            if batch_x.dim() == 3 and batch_x.size(1) == 1:
                batch_x = batch_x.squeeze(1)  # (B,1,T) -> (B,T)
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
        if batch_x.dim() == 3 and batch_x.size(1) == 1:
            batch_x = batch_x.squeeze(1)  # (B,1,T) -> (B,T)
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

# ========== 11. t-SNE 可视化 + 类内/类间距离（用 Log-Mel(+Δ/ΔΔ) 特征） ==========

# 1) 特征提取器：Log-Mel(dB) + Δ/ΔΔ，再做时间维均值/标准差池化到定长向量
sr = 16000
melspec = ta.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=512, hop_length=128, win_length=512,
    n_mels=64, center=True, power=2.0
)
to_db = ta.transforms.AmplitudeToDB(stype='power', top_db=80)

@torch.no_grad()
def extract_feat(wave_1d: torch.Tensor) -> np.ndarray:
    # 输入 (T,) 或 (1,T) -> 输出 1D (6F,)；已清理 NaN/Inf 并做 L2 归一化
    if wave_1d.dim() == 2:
        wave_1d = wave_1d.squeeze(0)

    S = melspec(wave_1d)         # power >= 0
    S_db = to_db(S)              # dB，裁顶后不会 -inf
    S_db = torch.nan_to_num(S_db, nan=-80.0, posinf=80.0, neginf=-80.0)

    d1 = ta.functional.compute_deltas(S_db)
    d2 = ta.functional.compute_deltas(d1)

    F = torch.cat([S_db, d1, d2], dim=0)                   # (3F, T')
    v = torch.cat([F.mean(dim=1), F.std(dim=1)], dim=0)    # (6F,)

    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    denom = v.norm(p=2)
    if torch.isfinite(denom) and denom > 0:
        v = v / denom
    return v.cpu().numpy()

# 2) 收集测试集全部特征与标签
features, labels_idx = [], []
with torch.no_grad():
    for x, y in test_loader:                # x:(B,T) 或 (B,1,T)
        for xi, yi in zip(x, y):
            features.append(extract_feat(xi.cpu()))
            labels_idx.append(int(yi))

features   = np.asarray(features)           # 期望 (N, D)
labels_idx = np.asarray(labels_idx)         # (N,)

# --- 终极对齐保险：强制 features 行=样本(N)，必要时转置 ---
if features.ndim > 2:
    features = features.reshape(features.shape[0], -1)

N = labels_idx.shape[0]
if features.shape[0] != N and features.shape[1] == N:
    print(f"[Fix] features 形状 {features.shape} 与样本数 {N} 不一致，尝试转置以对齐行=样本")
    features = features.T                   # 变为 (N, D)

if features.shape[0] != N:
    raise ValueError(f"features 的行数({features.shape[0]})必须等于样本数 N({N})；当前形状 {features.shape}。")

# --- 数值清理 + 过滤异常样本（注意用行掩码 [valid_mask, :]） ---
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
row_norm = np.linalg.norm(features, axis=1)          # (N,)
valid_mask = np.isfinite(row_norm) & (row_norm > 0)  # (N,)
if valid_mask.sum() < features.shape[0]:
    print(f"[Warn] 丢弃 {features.shape[0] - valid_mask.sum()} 个异常样本（NaN/Inf/全零）")
features   = features[valid_mask, :]
labels_idx = labels_idx[valid_mask]

# 方便显示：idx -> 文本标签
inv_label_map = {v: k for k, v in label2idx.items()}
labels_txt = np.array([inv_label_map[int(i)] for i in labels_idx])

# 3) 计算 类内/类间 距离（欧氏）及 Fisher 比 = inter / intra
feat2d = features if features.ndim == 2 else features.reshape(features.shape[0], -1)

classes = sorted(list(set(labels_txt)))     # 实际出现的类
centroids2d, intra_per_class = [], {}

for c in classes:
    Xc = feat2d[labels_txt == c]            # (Nc, D)
    if Xc.size == 0:
        continue
    Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
    mu = Xc.mean(axis=0, keepdims=True)     # (1, D)
    centroids2d.append(mu.squeeze(0))
    d = np.linalg.norm(Xc - mu, axis=1)
    d = d[np.isfinite(d)]
    intra_per_class[c] = float(d.mean()) if d.size > 0 else 0.0

centroids2d = np.stack(centroids2d, axis=0) if len(centroids2d) > 0 else np.zeros((0, feat2d.shape[1]))
centroids2d = np.nan_to_num(centroids2d, nan=0.0, posinf=0.0, neginf=0.0)

# —— 手写欧氏距离（更稳）——
if centroids2d.shape[0] >= 2 and centroids2d.shape[1] > 0:
    diff = centroids2d[:, None, :] - centroids2d[None, :, :]   # (K, K, D)
    Dmat = np.sqrt(np.sum(diff * diff, axis=2))                # (K, K)
else:
    Dmat = np.zeros((centroids2d.shape[0], centroids2d.shape[0]), dtype=np.float64)

# 汇总指标
mean_intra = float(np.mean(list(intra_per_class.values()))) if intra_per_class else 0.0
tri = np.triu_indices(Dmat.shape[0], k=1)
mean_inter = float(Dmat[tri].mean()) if tri[0].size > 0 else 0.0
fisher_ratio = mean_inter / (mean_intra + 1e-8) if mean_intra > 0 else 0.0

print("\n=== Distance Report (Log-Mel features) ===")
print(f"Mean intra-class distance : {mean_intra:.4f}")
print(f"Mean inter-class distance : {mean_inter:.4f}")
print(f"Fisher-style ratio (inter / intra): {fisher_ratio:.3f}")
print("Per-class intra / inter_mean / ratio:")
K = max(len(classes), 1)
for i, c in enumerate(classes):
    inter_c = (Dmat[i, :].sum() - Dmat[i, i]) / max(K - 1, 1)  # 去掉对角线
    print(f"  {c:20s}  intra={intra_per_class[c]:.4f}  inter_mean={inter_c:.4f}  ratio={inter_c/(intra_per_class[c]+1e-8):.2f}")

# —— t-SNE（合法 perplexity）——
if feat2d.shape[0] >= 3:  # t-SNE 至少要 3 个样本更稳
    perpl = int(max(2, min(30, feat2d.shape[0] - 1)))
    tsne = TSNE(n_components=2, perplexity=perpl, init='pca', learning_rate='auto', random_state=42)
    Z = tsne.fit_transform(feat2d)  # (N,2)

    plt.figure(figsize=(8, 6))
    for c in classes:
        idx = (labels_txt == c)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.85, label=c)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.title("t-SNE of Test Features (Log-Mel + Δ/ΔΔ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("[Warn] 样本数过少，跳过 t-SNE。")

# —— 质心间距离热力图 ——
if Dmat.shape[0] >= 2:
    plt.figure(figsize=(6, 5))
    plt.imshow(Dmat, cmap='Blues')
    plt.colorbar(label='Euclidean distance')
    plt.xticks(range(len(classes)), classes, rotation=60, ha='right')
    plt.yticks(range(len(classes)), classes)
    plt.title("Centroid-to-Centroid Distance (Euclidean)")
    plt.tight_layout()
    plt.show()