import torch
import torch.nn.functional as F
import numpy as np
import torchxrayvision as xrv
from utils import preprocess_for_model

# ---- PATCH torch.load to always allow full pickle (safe here because torchxrayvision is trusted) ----
_old_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_torch_load(*args, **kwargs)
torch.load = _patched_load

# Allow DenseNet class to be unpickled safely
torch.serialization.add_safe_globals([xrv.models.DenseNet])

LABELS = xrv.datasets.default_pathologies

def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

def predict(model, img_arr, device, score_mode="sigmoid"):
    x = preprocess_for_model(img_arr)
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    if score_mode == "softmax":
        probs = F.softmax(logits, dim=-1)
    else:
        probs = torch.sigmoid(logits)
    probs = probs.detach().cpu().numpy().flatten()
    logits = logits.detach().cpu()
    return probs, LABELS, logits, x

def build_cam(model, target_layer_name="features.norm5", device=None):
    from pytorch_grad_cam import GradCAM
    import torch.nn as nn

    # Navigate to the target layer
    target = model
    for part in target_layer_name.split("."):
        target = getattr(target, part)

    cam = GradCAM(model=model, target_layers=[target])
    return cam
