from pathlib import Path
from io import BytesIO
from typing import List

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import torch.backends.cudnn as cudnn

cudnn.enabled = False
cudnn.benchmark = False

MODEL_PATH = Path("models_torch/best_model_efficientnet_b0.pth")
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


model, CLASS_NAMES = load_model()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def predict_image(image: Image.Image, topk: int = 5):
    img_t = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)

    conf, pred_idx = torch.max(probs, dim=1)
    conf = float(conf.item())
    pred_label = CLASS_NAMES[pred_idx.item()]

    # top-k
    top_probs, top_indices = torch.topk(probs, k=min(topk, len(CLASS_NAMES)), dim=1)
    top_probs = top_probs.squeeze(0).cpu().numpy().tolist()
    top_indices = top_indices.squeeze(0).cpu().numpy().tolist()
    top_classes = [CLASS_NAMES[i] for i in top_indices]

    return pred_label, conf, list(zip(top_classes, top_probs))


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    file = request.files["image"]
    try:
        image = Image.open(BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    label, conf, topk = predict_image(image)
    return jsonify({
        "predicted_class": label,
        "confidence": conf,
        "top5": [
            {"class": cls, "score": float(score)} for cls, score in topk
        ],
    })


if __name__ == "__main__":
    print("Using device:", device)
    app.run(host="0.0.0.0", port=8000)
