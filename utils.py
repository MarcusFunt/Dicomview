"""Utility helpers for DICOM viewing."""

import math
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import pydicom
from pydicom.filereader import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom.pixel_data_handlers import pylibjpeg_handler, gdcm_handler
from PyQt6.QtGui import QImage


def jpeg2000_support_status() -> Tuple[bool, str]:
    """Return availability of JPEG2000 handlers and a summary string."""
    details, ok = [], False
    try:
        if pylibjpeg_handler.is_available():
            ok = True
            details.append("pylibjpeg-openjpeg: OK")
    except Exception as e:  # pragma: no cover - diagnostic only
        details.append(f"pylibjpeg handler error: {e}")
    try:
        if gdcm_handler.is_available():
            ok = True
            details.append("gdcm: OK")
    except Exception as e:  # pragma: no cover - diagnostic only
        details.append(f"gdcm handler error: {e}")
    return ok, " | ".join(details) if details else "no details"


def is_dicom_file(path: str) -> bool:
    """Heuristically determine if a path points to a DICOM file."""
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
            if len(pre) >= 132 and pre[128:132] == b"DICM":
                return True
    except Exception:
        pass
    try:
        ds = dcmread(path, stop_before_pixels=True, force=True)
        return hasattr(ds, "SOPClassUID") or hasattr(ds, "PatientID")
    except Exception:
        return False


def try_open_image_any(path: str) -> Optional[Image.Image]:
    """Attempt to open a generic image file, returning ``None`` on failure."""
    try:
        im = Image.open(path)
        im.load()
        return im
    except Exception:
        return None


def series_key_from_ds(ds: pydicom.dataset.FileDataset) -> str:
    return getattr(ds, "SeriesInstanceUID", None) or (
        f"{getattr(ds, 'StudyInstanceUID', '')}"
        f"|{getattr(ds, 'PatientID', '')}|{getattr(ds, 'Modality', '')}"
    )


def sort_key_from_ds(ds: pydicom.dataset.FileDataset) -> Tuple:
    ipp = getattr(ds, "ImagePositionPatient", None)
    z = float(ipp[2]) if isinstance(ipp, (list, tuple)) and len(ipp) >= 3 else math.inf
    return (z, getattr(ds, "InstanceNumber", 1e12))


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max == a_min:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - a_min) / (a_max - a_min)
    return np.clip((scaled * 255).round(), 0, 255).astype(np.uint8)


def dicom_to_ndarray(ds: pydicom.dataset.FileDataset, frame_index: int = 0) -> np.ndarray:
    arr = ds.pixel_array
    if arr.ndim == 3 and getattr(ds, "NumberOfFrames", None):
        arr = arr[frame_index]
    elif arr.ndim == 4:
        arr = arr[frame_index]
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr.astype(np.float32) * slope + intercept
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2").upper()
    if arr.ndim == 2:
        arr = normalize_to_uint8(arr)
        if photometric == "MONOCHROME1":
            arr = 255 - arr
        return arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return normalize_to_uint8(arr)
    return normalize_to_uint8(arr.squeeze())


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a numpy array to a ``QImage`` ensuring a contiguous buffer."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.tobytes(), w, h, arr.strides[0], QImage.Format.Format_Grayscale8)
        return qimg.copy()
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        qimg = QImage(arr.tobytes(), w, h, arr.strides[0], QImage.Format.Format_RGB888)
        return qimg.copy()
    return numpy_to_qimage(normalize_to_uint8(arr.squeeze()))
