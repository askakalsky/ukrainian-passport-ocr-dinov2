"""Experiment with a frozen Meta DINOv2 encoder and a single-field detector."""

import hashlib
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "data/external_cache/dinov2"
WEIGHTS = VENDOR.with_name("dinov2_vits14_pretrain.pth")
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
SIZE = 448
GRID = SIZE // 14
FILL = (114, 114, 114)


def load_encoder(device):
    revision = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(VENDOR), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != REVISION or dirty:
        raise ValueError("Expected the pinned, unmodified Meta DINOv2 checkout")
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("DINOv2 starting weights do not match the recorded SHA256")
    sys.path.insert(0, str(VENDOR))
    try:
        from dinov2.hub.backbones import dinov2_vits14
        model = dinov2_vits14(pretrained=False)
    finally:
        sys.path.remove(str(VENDOR))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False).to(device)


def letterbox(image):
    h, w = image.shape[:2]
    ratio = SIZE / max(h, w)
    width, height = max(1, round(w * ratio)), max(1, round(h * ratio))
    offset = ((SIZE - width) // 2, (SIZE - height) // 2)
    x, y = offset
    canvas = np.full((SIZE, SIZE, 3), FILL, dtype=np.uint8)
    canvas[y:y+height, x:x+width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return canvas, np.array([width/w, height/h] * 2), np.array(offset * 2)


@torch.no_grad()
def features(encoder, images, device):
    x = torch.from_numpy(np.stack(images)).to(device).permute(0, 3, 1, 2).flip(1).float() / 255
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        layers = encoder.get_intermediate_layers((x - mean) / std, n=[5, 11], reshape=True)
    return torch.cat(layers, dim=1)


class BoxHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(768, 96, 1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 5, 1),
        )


def grid(output):
    axis = (torch.arange(GRID, device=output.device, dtype=output.dtype) + .5) / GRID
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([x, y, x, y])


def best_box(output):
    offsets = output[:, 1:] / 4 + grid(output)
    probabilities = output[:, 0].flatten(1).softmax(dim=1)
    score, index = probabilities.max(dim=1)
    box = offsets.flatten(2).gather(2, index[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    return box, score


class DINOv2Locator:
    def __init__(self, path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["source_revision"] != REVISION or state["initial_sha256"] != WEIGHTS_SHA256:
            raise ValueError("Checkpoint requires a different DINOv2 encoder")
        self.encoder = load_encoder(self.device)
        self.head = BoxHead().to(self.device).eval()
        self.head.load_state_dict(state["model_state"])

    @torch.inference_mode()
    def locate(self, image):
        canvas, scale, shift = letterbox(image)
        box, score = best_box(self.head(features(self.encoder, [canvas], self.device).float()))
        box = (box[0].cpu().numpy() * SIZE - shift) / scale
        h, w = image.shape[:2]
        if not np.isfinite(box).all():
            return None, 0.
        x, y, r, b = box.clip(0, [w, h, w, h])
        if r <= x or b <= y:
            return None, float(score[0])
        return [math.floor(x), math.floor(y), math.ceil(r), math.ceil(b)], float(score[0])
