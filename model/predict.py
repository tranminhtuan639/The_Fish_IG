# model/predict.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# py model/predict.py data/raw/arowana/check2.jpg
# ─── Config ───────────────────────────────────────────────
MODEL_PATH  = "model/checkpoints/best_phase2.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD   = 0.60  # Dưới ngưỡng này → không nhận diện được

# ─── Transform (giống val_transform khi train) ────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Load model ───────────────────────────────────────────
def load_model():
    checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
    classes     = checkpoint["classes"]
    num_classes = len(classes)

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, classes

# ─── Predict 1 ảnh ────────────────────────────────────────
def predict(image_input, model=None, classes=None):
    """
    Args:
        image_input: đường dẫn file (str) hoặc PIL Image
        model, classes: truyền vào nếu đã load sẵn (tránh load lại mỗi lần)
    Returns:
        dict: {
            "status":     "ok" | "uncertain",
            "species":    "arowana" | None,
            "confidence": 0.96 | None,
            "message":    str
        }
    """
    if model is None or classes is None:
        model, classes = load_model()

    # Load ảnh
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")  # PIL Image từ Streamlit

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs     = model(img_tensor)
        probs       = torch.softmax(outputs, dim=1)
        confidence, idx = probs.max(1)

    confidence  = confidence.item()
    species     = classes[idx.item()]

    if confidence < THRESHOLD:
        return {
            "status":     "uncertain",
            "species":    None,
            "confidence": confidence,
            "message":    "Không nhận diện được loài cá này. Vui lòng thử ảnh khác rõ hơn."
        }

    return {
        "status":     "ok",
        "species":    species,
        "confidence": round(confidence, 4),
        "message":    f"Đây là {species} (độ tin cậy: {confidence:.1%})"
    }


# ─── Test nhanh khi chạy trực tiếp ───────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: py predict.py <đường_dẫn_ảnh>")
        print("Ví dụ: py predict.py data/raw/arowana/check.jpg")
        sys.exit(1)

    image_path  = sys.argv[1]
    model, classes = load_model()
    result      = predict(image_path, model, classes)

    print(f"\n🐟 Kết quả nhận diện:")
    print(f"   Status     : {result['status']}")
    print(f"   Species    : {result['species']}")
    print(f"   Confidence : {result['confidence']}")
    print(f"   Message    : {result['message']}")