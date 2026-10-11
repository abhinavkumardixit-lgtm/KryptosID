"""
Script to download real and fake benchmark face samples from Hugging Face
(Indian-deepfake-face + FaceForensics++ samples), organize into data directories,
generate identity/generator disjoint splits, and fine-tune KryptosIDNet.
"""

import os
import sys
import json
import urllib.request
from typing import List, Tuple
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "mrxaman19/Indian-deepfake-face"
DATA_DIR = "data"


def fetch_benchmark_dataset(num_samples: int = 100) -> Tuple[List[str], List[str]]:
    """Downloads balanced real and fake images from Hugging Face."""
    os.makedirs(os.path.join(DATA_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "fake"), exist_ok=True)

    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)

    real_files = [s.rfilename for s in info.siblings if s.rfilename.startswith("real/")][:num_samples]
    fake_files = [s.rfilename for s in info.siblings if s.rfilename.startswith("fake/")][:num_samples]

    print(f"Downloading {len(real_files)} real images...")
    real_paths = []
    for i, r_file in enumerate(real_files):
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=r_file,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        real_paths.append(local_path)
        if (i + 1) % 25 == 0:
            print(f"  [Real] Downloaded {i + 1}/{len(real_files)}")

    print(f"Downloading {len(fake_files)} fake images...")
    fake_paths = []
    for i, f_file in enumerate(fake_files):
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f_file,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
        fake_paths.append(local_path)
        if (i + 1) % 25 == 0:
            print(f"  [Fake] Downloaded {i + 1}/{len(fake_files)}")

    print(f"Successfully downloaded {len(real_paths)} real and {len(fake_paths)} fake images to {DATA_DIR}.")
    return real_paths, fake_paths


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    fetch_benchmark_dataset(num_samples=n)
