import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, csv
from tqdm import tqdm
from model.PCT_Net import CNN_Transformer_AudioClassifier
from processed_librosa.data_loader import get_data_loaders

torch.backends.cudnn.benchmark = True

# === 1. 标签映射 ===
train_df = pd.read_csv('divided_data/train.csv')
label_list = sorted(train_df['label'].unique())
label2idx = {label: idx for idx, label in enumerate(label_list)}

# === 2. 训练设置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
num_epochs = 10
batch_size = 32
learning_rate = 1e-4

# === 3. 数据加载器 ===
train_loader, val_loader, test_loader = get_data_loaders(
    './divided_data/train.csv',
    './divided_data/val.csv',
    './divided_data/test.csv',
    batch_size=batch_size,
    label2idx=label2idx
)

# === 4. 消融实验配置 ===
ABLATIONS = [
    # ("torchaudio-noaug", dict(front_end="torchaudio", use_specaug=False)),
    # ("panns+aug", dict(front_end="panns", use_specaug=True)),
    # ("torchaudio+aug", dict(front_end="torchaudio", use_specaug=True)),
    # ("panns-noaug", dict(front_end="panns", use_specaug=False)),
]

results_rows = []
csv_path = "ablation_results.csv"
write_header = not os.path.exists(csv_path)

for tag, cfg in ABLATIONS:
    print(f"\n[Running Ablation] {tag} | Config: {cfg}")

    model = CNN_Transformer_AudioClassifier(
        sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
        fmin=50, fmax=8000, classes_num=len(label2idx),
        front_end=cfg["front_end"],
        use_specaug=cfg["use_specaug"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_path = f"best_{tag}.pth"

    # === 5. 训练 ===
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"{tag} Epoch {epoch+1}", ncols=80):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # === 验证 ===
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                outputs = model(val_x)
                _, pred = torch.max(outputs, 1)
                correct += (pred == val_y).sum().item()
                total += val_y.size(0)
        val_acc = correct / total
        print(f"[Val] Epoch {epoch+1}: val_acc = {val_acc:.4f} | train_loss = {avg_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Best model saved to: {best_model_path}")

    # === 6. 测试 ===
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            outputs = model(test_x)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            gts.extend(test_y.cpu().numpy())
    test_acc = (np.array(preds) == np.array(gts)).mean()
    print(f"[Test] {tag} Accuracy = {test_acc:.4f}")

    # === 7. 结果写入 CSV ===
    row = {
        "tag": tag,
        "front_end": cfg["front_end"],
        "use_specaug": int(cfg["use_specaug"]),
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4)
    }
    results_rows.append(row)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
            write_header = False
        writer.writerow(row)

# === 8. 图表绘制（读取历史记录+当前记录）===
if os.path.exists(csv_path):
    df_all = pd.read_csv(csv_path)
    df_all = df_all.drop_duplicates(subset=["tag"], keep="last")

    print("\n[全实验汇总]")
    print(df_all)

    # 画图
    plt.figure(figsize=(8, 5))
    plt.bar(df_all["tag"], df_all["test_acc"], width=0.6, color='steelblue')
    for i, (x, y) in enumerate(zip(df_all["tag"], df_all["test_acc"])):
        plt.text(i, y + 0.01, f"{y:.3f}", ha='center', va='bottom')
    plt.ylabel("Test Accuracy")
    plt.title("Ablation on FrontEnd & SpecAug")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()