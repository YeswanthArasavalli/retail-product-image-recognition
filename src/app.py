from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st
import torch.backends.cudnn as cudnn

# ---------- CONFIG ----------
MODEL_PATH = Path("models_torch/best_model_efficientnet_b0.pth")
IMG_SIZE = 224

# Streamlit Cloud will most likely use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disable cuDNN to avoid any weird conv issues (safe even on CPU)
cudnn.enabled = False
cudnn.benchmark = False


@st.cache_resource
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


def predict(image):
    img_t = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)

    conf, pred_idx = torch.max(probs, dim=1)
    conf = float(conf.item())
    pred_idx = int(pred_idx.item())
    pred_label = CLASS_NAMES[pred_idx]

    # top-5
    top_probs, top_indices = torch.topk(probs, k=min(5, len(CLASS_NAMES)), dim=1)
    top_probs = top_probs.squeeze(0).cpu().numpy().tolist()
    top_indices = top_indices.squeeze(0).cpu().numpy().tolist()
    top_classes = [CLASS_NAMES[i] for i in top_indices]

    return pred_label, conf, list(zip(top_classes, top_probs))


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Retail Product Image Recognition", layout="centered")

st.title("🛒 Retail Product Image Recognition")
st.caption("Upload a product image to classify it into one of ~200 SKU categories.")

uploaded = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Predicting..."):
        label, conf, topk = predict(image)

    st.subheader("Top-1 Prediction")
    st.write(f"**Class:** `{label}`")
    st.write(f"**Confidence:** `{conf*100:.2f}%`")

    st.subheader("Top-5 Predictions")
    for cls, score in topk:
        st.write(f"- `{cls}` — {score*100:.2f}%")
else:
    st.info("Upload an image to get a prediction.")
