# model/evaluate.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)
from collections import Counter

# ─── Config ───────────────────────────────────────────────
DATA_DIR   = "data/processed/test"
MODEL_PATH = "model/checkpoints/best_phase2.pth"
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Transform (giống val khi train) ──────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Load model ───────────────────────────────────────────
def load_model(num_classes):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, checkpoint["classes"]

# ─── Evaluate ─────────────────────────────────────────────
def evaluate():
    print(f"📂 Test set: {DATA_DIR}")
    print(f"🔥 Device  : {DEVICE}\n")

    # Dataset
    test_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2
    )

    # Load model
    model, classes = load_model(num_classes=len(test_dataset.classes))
    print(f"✅ Loaded model: {MODEL_PATH}")
    print(f"   Classes: {classes}\n")

    # Kiểm tra thứ tự class khớp không
    assert list(test_dataset.classes) == list(classes), \
        "❌ Class order mismatch giữa dataset và checkpoint!"

    # ── Inference ─────────────────────────────────────────
    all_preds  = []
    all_labels = []
    all_confs  = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    # ── Số ảnh test mỗi loài ──────────────────────────────
    print("📊 Số ảnh test mỗi loài:")
    count = Counter(all_labels)
    for i, name in enumerate(classes):
        print(f"  {name:12s}: {count[i]} ảnh")

    # ── Overall metrics ───────────────────────────────────
    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print(f"\n{'='*50}")
    print(f"  Accuracy        : {accuracy:.4f} ({accuracy:.1%})")
    print(f"  F1 Macro        : {f1_macro:.4f}")
    print(f"  F1 Weighted     : {f1_weighted:.4f}")
    print(f"  Avg Confidence  : {all_confs.mean():.4f}")
    print(f"{'='*50}\n")

    # ── Per-class report ──────────────────────────────────
    print("📈 Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names = classes,
        digits       = 4
    ))

    # ── Confidence per class ──────────────────────────────
    print("🎯 Average Confidence mỗi loài:")
    for i, name in enumerate(classes):
        mask     = all_labels == i
        avg_conf = all_confs[mask].mean() if mask.sum() > 0 else 0
        print(f"  {name:12s}: {avg_conf:.4f}")

    # ── Confusion Matrix ──────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = classes
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — Fish Classification", fontsize=14, pad=15)
    plt.tight_layout()

    save_path = "model/checkpoints/confusion_matrix.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n💾 Saved confusion matrix: {save_path}")
    plt.show()

if __name__ == "__main__":
    evaluate()