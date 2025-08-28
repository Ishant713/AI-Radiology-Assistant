import streamlit as st
import numpy as np
import torch
import cv2
from inference import load_model, predict, build_cam
from utils import load_image_any, to_display, overlay_heatmap

st.set_page_config(page_title="AI Assistant for Radiology Images", layout="wide")
st.title("AI Assistant for Radiology Images")
st.caption("Educational demo. Not for clinical use.")

# -----------------------------
# Helper function: crop + resize to square
# -----------------------------
def preprocess_square(img_arr, size=224):
    img_arr = img_arr.astype(np.float32)
    h, w = img_arr.shape[:2]
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    img_arr = img_arr[start_h:start_h + min_dim, start_w:start_w + min_dim]
    img_arr = cv2.resize(img_arr, (size, size))
    return img_arr

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Upload")
    file = st.file_uploader("JPG/PNG/DICOM (.dcm)", type=["jpg","jpeg","png","dcm"])
    thresh = st.slider("Probability threshold", 0.05, 0.95, 0.3, 0.05)
    cam_layer = st.text_input("Grad-CAM target layer", "features.norm5")
    cam_on = st.toggle("Show Grad-CAM", value=True)
    st.divider()
    st.header("Model")
    device_opt = st.selectbox("Device", ["auto","cpu","cuda"], index=0)
    score_mode = st.selectbox("Score mode", ["sigmoid","softmax"], index=0)

# -----------------------------
# Load model once
# -----------------------------
if "model" not in st.session_state:
    st.session_state["model"] = load_model()

if file is None:
    st.info("Upload a chest X-ray image to begin.")
    st.stop()

# -----------------------------
# Read file and preprocess
# -----------------------------
file.seek(0)
data = file.read()
img_arr, info = load_image_any(data, file.name)

# ✅ Crop + resize to square for model
img_arr = preprocess_square(img_arr, size=224)

# Debug info
st.write("DEBUG stats → shape:", img_arr.shape, "min:", np.min(img_arr), "max:", np.max(img_arr), "mean:", np.mean(img_arr))

disp = to_display(img_arr)
if isinstance(disp, np.ndarray):
    if disp.dtype != np.uint8:
        disp = (255 * (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)).astype(np.uint8)
    st.image(disp, caption="Input", width=500)
else:
    st.image(disp, caption="Input", width=500)

# -----------------------------
# Model + Device
# -----------------------------
model = st.session_state["model"]
device = torch.device("cuda" if (device_opt == "cuda" or (device_opt == "auto" and torch.cuda.is_available())) else "cpu")
model.to(device).eval()

# -----------------------------
# Predictions
# -----------------------------
probs, labels, logits, input_tensor = predict(model, img_arr, device, score_mode)

import pandas as pd
df = pd.DataFrame({"label": labels, "probability": probs})
df = df.sort_values("probability", ascending=False)

st.subheader("Predictions")
st.dataframe(df, use_container_width=True)

# -----------------------------
# Grad-CAM
# -----------------------------
if cam_on:
    cam = build_cam(model, target_layer_name=cam_layer, device=device)
    targets = None
    if logits.shape[-1] > 0:
        top_idx = int(torch.argmax(logits).item())
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(top_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    heatmap = overlay_heatmap(disp, grayscale_cam)
    st.subheader("Grad-CAM")
    st.image(heatmap, width=500)

# -----------------------------
# Metadata
# -----------------------------
with st.expander("Image metadata"):
    st.json(info)
