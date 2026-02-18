import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SpatialImageDataset(Dataset):
    def __init__(self, data_dir: Path, metadata_file: str = "metadata.json"):
        """Initialize the VolumetricDataset using PNG files and JSON metadata."""

        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / metadata_file

        if not self.metadata_path.is_file():
            raise ValueError(f"Metadata file not found at {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            payload: Dict[str, List[Dict[str, object]]] = json.load(handle)

        self.entries = payload.get("images", [])

        if not self.entries:
            raise ValueError(f"No image metadata entries found in {self.metadata_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return image tensor and associated camera parameters."""

        entry = self.entries[idx]
        image_path = self.data_dir / entry["file_name"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Image file {image_path} is missing")

        image = Image.open(image_path).convert("RGBA")
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)

        position = torch.tensor(entry["position"], dtype=torch.float32)
        rotation = torch.tensor(entry["rotation"], dtype=torch.float32)
        fov = float(entry["fov"])

        return image_tensor, position, rotation, fov
