import os
from typing import Dict, List, Optional

import numpy as np
import pydicom

from utils import is_dicom_file, series_key_from_ds, sort_key_from_ds, dicom_to_ndarray


def discover_series(folder: str) -> Dict[str, Dict]:
    """Scan ``folder`` for DICOM files grouped by series."""
    series_data: Dict[str, Dict] = {}
    for root, _, files in os.walk(folder):
        for fn in files:
            path = os.path.join(root, fn)
            if not is_dicom_file(path):
                continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            key = series_key_from_ds(ds)
            series_data.setdefault(key, {"paths": [], "meta": ds})
            series_data[key]["paths"].append(path)

    # sort slices
    for key, sd in series_data.items():
        paths = sd["paths"]
        sort_list = []
        for p in paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            except Exception:
                continue
            sort_list.append((sort_key_from_ds(ds), p))
        sort_list.sort(key=lambda x: x[0])
        sd["paths"] = [p for _, p in sort_list]
    return series_data


def load_volume(paths: List[str]) -> Optional[np.ndarray]:
    """Load a stack of DICOM files into a 3D numpy array."""
    vol = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p, force=True)
            vol.append(dicom_to_ndarray(ds))
        except Exception:
            continue
    return np.stack(vol, axis=0) if vol else None
