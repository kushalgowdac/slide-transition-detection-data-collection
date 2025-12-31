"""Utility helpers for IO, hashing and basic image ops."""
from pathlib import Path
import hashlib
import cv2
import numpy as np


def compute_md5_bytes(img_bytes: bytes) -> str:
    return hashlib.md5(img_bytes).hexdigest()


def safe_imencode(img, ext='.jpg') -> bytes:
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError('imencode failed')
    return buf.tobytes()


def read_image(path: str, gray: bool = False):
    if gray:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
