# model/train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from collections import Counter

# ─── Seed ─────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# ─── Config ───────────────────────────────────────────────
DATA_DIR       = "data/processed"
SAVE_DIR       = "model/checkpoints"
NUM_CLASSES    = 4
BATCH_SIZE     = 32
EPOCHS         = 50
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNFREEZE_EPOCH = 10

print("🔥 Device:", DEVICE)
if torch.cuda.is_available():
    print("   GPU:", torch.cuda.get_device_name(0))

# ─── Focal Loss ───────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Thay CrossEntropy, phạt nặng hơn với mẫu khó.
    gamma=2.0: mẫu dễ (pt cao) → weight nhỏ, mẫu khó (pt thấp) → weight lớn.
    weight=None vì WeightedRandomSampler đã cân bằng class rồi.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ─── Early Stopping ───────────────────────────────────────
class EarlyStopping:
    """
    Dừng train sớm nếu val_acc không cải thiện sau `patience` epoch.
    Reset khi chuyển phase để tránh trigger sớm sau khi unfreeze backbone.
    """
    def __init__(self, patience=7):
        self.patience  = patience
        self.counter   = 0
        self.best_acc  = 0.0
        self.triggered = False

    def step(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            print(f"  ⏳ EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True

# ─── Transforms ───────────────────────────────────────────
train_transform = transforms.Compose([
    # RandomResizedCrop thay RandomCrop: giữ 85-100% ảnh gốc, tránh mất đuôi/vây cá
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    # Rotation nhỏ (10°) vì cá bơi ngang, không cần xoay nhiều
    transforms.RandomRotation(10),
    # ColorJitter nhẹ: giữ màu đặc trưng (vảy arowana đỏ, oscar cam,...)
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    # Val/test không augment, chỉ resize về đúng kích thước model
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Dataset ──────────────────────────────────────────────
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), transform=train_transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"), transform=val_transform
)

# ─── WeightedRandomSampler ────────────────────────────────
# Dataset mất cân bằng (flowerhorn 102 vs snakehead 251)
# Sampler tăng tần suất sample loài ít → model không bias về loài nhiều
labels      = [label for _, label in train_dataset.samples]
class_count = Counter(labels)

print("\n📊 Số ảnh mỗi loài (train):")
for idx, name in enumerate(train_dataset.classes):
    print(f"  {name}: {class_count[idx]} ảnh")

class_weights  = [1.0 / class_count[i] for i in range(NUM_CLASSES)]
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    sampler=sampler, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=True
)

# ─── Model ────────────────────────────────────────────────
# EfficientNet-B0 pretrained ImageNet: đã biết texture, shape, edge, màu sắc
# → chỉ cần fine-tune, không train từ đầu
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Thay classifier TRƯỚC khi freeze — tránh nhầm lẫn requires_grad
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)

# Freeze toàn bộ backbone — Phase 1 chỉ train classifier
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(DEVICE)

# ─── Loss & Scaler ────────────────────────────────────────
criterion = FocalLoss(gamma=2.0)

# GradScaler cho AMP: scale loss để tránh underflow với FP16
scaler = GradScaler()

# ─── Optimizer theo phase ─────────────────────────────────
def get_optimizer(phase):
    if phase == 1:
        # Phase 1: chỉ train classifier, LR cao để hội tụ nhanh
        return optim.AdamW(
            model.classifier.parameters(),
            lr=1e-3, weight_decay=1e-4
        )
    else:
        # Phase 2: differential LR — backbone nhỏ (1e-5) tránh phá pretrained
        #                           classifier lớn hơn (1e-4) để tinh chỉnh
        return optim.AdamW([
            {"params": model.features[-3:].parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(),    "lr": 1e-4},
        ], weight_decay=1e-4)

# ─── Train loop ───────────────────────────────────────────
def run_epoch(loader, optimizer=None, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()
                # AMP: tự cast FP32 → FP16 trong forward, tận dụng Tensor Core RTX 4050
                with autocast():
                    outputs = model(imgs)
                    loss    = criterion(outputs, labels)
                # scaler tránh gradient underflow khi dùng FP16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Val không cần AMP, chạy FP32 cho stable
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total

# ─── Per-class accuracy ───────────────────────────────────
def per_class_accuracy(loader, classes):
    """
    In accuracy từng loài — quan trọng cho RAG:
    model 85% overall có thể đang 95% arowana + 40% flowerhorn
    → cần biết để điều chỉnh confidence threshold từng loài
    """
    model.eval()
    correct_per_class = Counter()
    total_per_class   = Counter()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            for p, t in zip(preds, labels):
                total_per_class[t.item()]   += 1
                if p == t:
                    correct_per_class[t.item()] += 1

    print("\n📈 Per-class accuracy (val):")
    for i, name in enumerate(classes):
        acc = correct_per_class[i] / total_per_class[i] if total_per_class[i] else 0
        print(f"  {name:12s}: {acc:.4f} ({correct_per_class[i]}/{total_per_class[i]})")

# ─── Main ─────────────────────────────────────────────────
print(f"\n🚀 Training trên : {DEVICE}")
print(f"   Classes       : {train_dataset.classes}")
print(f"   Epochs        : {EPOCHS}")
print(f"   Unfreeze tại  : Epoch {UNFREEZE_EPOCH}")
print(f"   Seed          : {SEED}\n")

os.makedirs(SAVE_DIR, exist_ok=True)

optimizer      = get_optimizer(phase=1)
scheduler      = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)
early_stopping = EarlyStopping(patience=7)
best_val_acc   = 0.0
current_phase  = 1

for epoch in range(1, EPOCHS + 1):

    # ── Chuyển Phase 2 tại UNFREEZE_EPOCH ────────────────
    if epoch == UNFREEZE_EPOCH and current_phase == 1:
        print(f"\n🔓 Epoch {epoch}: Unfreeze 3 block cuối — chuyển Phase 2\n")

        for param in model.features[-3:].parameters():
            param.requires_grad = True

        optimizer = get_optimizer(phase=2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )

        # Reset EarlyStopping: model có thể dip nhẹ ngay sau khi unfreeze
        early_stopping          = EarlyStopping(patience=7)
        early_stopping.best_acc = best_val_acc

        # Lưu Phase 1 để rollback nếu Phase 2 phá model
        torch.save({
            "epoch":       epoch - 1,
            "model_state": model.state_dict(),
            "classes":     train_dataset.classes,
            "val_acc":     best_val_acc,
        }, os.path.join(SAVE_DIR, "best_phase1.pth"))
        print(f"  💾 Saved best_phase1.pth (val_acc={best_val_acc:.4f})")

        current_phase = 2

    train_loss, train_acc = run_epoch(train_loader, optimizer=optimizer, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   optimizer=None,      training=False)
    scheduler.step(val_acc)

    phase_tag = "P1" if current_phase == 1 else "P2"
    print(f"[{phase_tag}] Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_name    = f"best_phase{current_phase}.pth"
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "classes":     train_dataset.classes,
            "val_acc":     val_acc,
        }, os.path.join(SAVE_DIR, save_name))
        print(f"  ✅ Saved {save_name} (val_acc={val_acc:.4f})")

    early_stopping.step(val_acc)
    if early_stopping.triggered:
        print(f"\n🛑 Early stopping tại epoch {epoch}")
        break

print(f"\n🏁 Done! Best val accuracy: {best_val_acc:.4f}")
per_class_accuracy(val_loader, train_dataset.classes)