import numpy as np
from PIL import Image
import pydicom
import io
import cv2

def load_image_any(data_bytes, filename):
    name = filename.lower()
    if name.endswith(".dcm"):
        ds = pydicom.dcmread(io.BytesIO(data_bytes))
        arr = ds.pixel_array.astype("float32")
        arr = apply_windowing(ds, arr)
        info = {
            "modality": str(getattr(ds, "Modality", "")),
            "patient_id": str(getattr(ds, "PatientID", "")),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "rows": int(ds.Rows),
            "cols": int(ds.Columns)
        }
        arr = normalize_to_uint8(arr)
        return arr, info
    else:
        im = Image.open(io.BytesIO(data_bytes)).convert("L")  # force grayscale
        arr = np.array(im)
        info = {"mode": "L", "shape": list(arr.shape)}
        return arr, info

def apply_windowing(ds, arr):
    try:
        center = ds.WindowCenter
        width = ds.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = float(center[0])
        else:
            center = float(center)
        if isinstance(width, pydicom.multival.MultiValue):
            width = float(width[0])
        else:
            width = float(width)
        low = center - width / 2
        high = center + width / 2
        arr = np.clip(arr, low, high)
    except Exception:
        pass
    return arr

def normalize_to_uint8(arr):
    arr = arr.astype("float32")
    mn, mx = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    if mx - mn < 1e-5:
        mn, mx = arr.min(), arr.max()
    arr = np.clip((arr - mn) / (mx - mn + 1e-8), 0, 1)
    arr = (arr * 255).astype("uint8")
    return arr

def preprocess_for_model(img_arr):
    # ensure image is grayscale and normalized
    if img_arr.ndim == 3:   # RGB
        img_arr = np.mean(img_arr, axis=2)
    img_arr = img_arr.astype(np.float32)
    img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))  # normalize [0,1]
    img_arr = img_arr * 255
    img_arr = img_arr[None, :, :]  # add channel dim
    return img_arr

def to_display(arr):
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

def overlay_heatmap(img_gray_or_rgb, cam_gray):
    if cam_gray.min() < 0 or cam_gray.max() > 1:
        cam_gray = (cam_gray - cam_gray.min()) / (cam_gray.max() - cam_gray.min() + 1e-8)
    if img_gray_or_rgb.ndim == 2:
        base = cv2.cvtColor(img_gray_or_rgb, cv2.COLOR_GRAY2RGB)
    else:
        base = img_gray_or_rgb
    cam_resized = cv2.resize(cam_gray.astype("float32"), (base.shape[1], base.shape[0]))
    heat = cv2.applyColorMap((cam_resized * 255).astype("uint8"), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (0.5 * base + 0.5 * heat).astype("uint8")
    return out
