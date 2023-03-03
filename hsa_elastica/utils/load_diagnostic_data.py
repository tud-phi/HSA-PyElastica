import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List


def load_diagnostic_data(log_dir: str) -> List[Dict]:
    with open(str(Path(log_dir) / "diagnostic_data.pkl"), "rb") as f:
        data = pickle.load(f)

    return data
