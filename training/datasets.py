"""PyTorch Dataset implementation for KryptosID splits and forensic pipeline."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from training.augment import ForensicAugmentationPipeline
from training.splits import SplitManifest, verify_split_manifest


class ForensicDataset(Dataset):
    """Dataset wrapper for KryptosID sealed manifests and augmentation pipeline."""

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        generate_mock_on_missing: bool = True,
    ) -> None:
        """
        Args:
            manifest_path: Path to sealed manifest JSON.
            split: One of 'train', 'val', 'test'.
            transform: Optional transform pipeline. If None, uses ForensicAugmentationPipeline.
            generate_mock_on_missing: If True, generates a synthetic sample image when file_path is missing.
        """
        self.manifest: SplitManifest = verify_split_manifest(manifest_path)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

        self.split_name = split
        self.samples: List[Dict[str, Any]] = getattr(self.manifest, split)
        self.generate_mock_on_missing = generate_mock_on_missing

        if transform is not None:
            self.transform = transform
        else:
            self.transform = ForensicAugmentationPipeline(is_training=(split == "train"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Fetch and preprocess sample.
        
        Returns:
            Tuple of:
            - spatial_tensor: [3, 224, 224] ImageNet-normalized float32 tensor
            - spectral_gray: [1, 224, 224] un-normalized float32 tensor in [0, 1]
            - label: float32 tensor [1] (0.0 genuine, 1.0 manipulated)
            - sample_id: string ID
        """
        item = self.samples[idx]
        file_path = item["file_path"]
        label_val = float(item["label"])
        sample_id = item["sample_id"]

        if os.path.exists(file_path):
            image = Image.open(file_path).convert("RGB")
        elif self.generate_mock_on_missing:
            # Generate deterministic synthetic image from hash of sample_id for mock testing
            seed = int(abs(hash(sample_id))) % (2**31)
            rng = np.random.RandomState(seed)
            img_arr = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(img_arr)
        else:
            raise FileNotFoundError(f"Sample image not found: {file_path}")

        spatial, spectral = self.transform(image)
        label_tensor = torch.tensor([label_val], dtype=torch.float32)

        return spatial, spectral, label_tensor, sample_id
