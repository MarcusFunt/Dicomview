import subprocess
from typing import Optional

import numpy as np

from utils import normalize_to_uint8


def normalize_volume(volume: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize ``volume`` to ``uint8`` if not ``None``."""
    if volume is None:
        return None
    return normalize_to_uint8(volume)


def run_nppy(input_folder: str, output_folder: str):
    """Run the external ``nppy`` pre-processing command."""
    subprocess.run(["nppy", "-i", input_folder, "-o", output_folder], check=True)
