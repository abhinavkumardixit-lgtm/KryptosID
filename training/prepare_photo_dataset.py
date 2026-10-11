"""
Extract real vs. AI-generated face photo dataset from parquet.
Combines real human portrait photographs with AI-synthesized faces.
Produces balanced photo samples and integrates with FaceForensics++ video frames.
"""

import os
import io
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from training.splits import DatasetSample, SplitManifest, create_identity_disjoint_split

REPO_ID = "desireemcv/face-real-vs-fake"
PARQUET_FILE = "data/validation-00000-of-00001.parquet"
OUTPUT_DIR = "data/photos"


def extract_photo_dataset(num_real: int = 120, num_fake: int = 120) -> List[DatasetSample]:
    """Downloads parquet and extracts balanced real human photos and AI-generated portraits."""
    os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)

    print("Fetching photo dataset parquet from Hugging Face...")
    parquet_path = hf_hub_download(repo_id=REPO_ID, filename=PARQUET_FILE, repo_type="dataset")

    df = pq.read_table(parquet_path).to_pandas()
    print(f"Total rows in parquet: {len(df)}")

    samples: List[DatasetSample] = []
    real_count = 0
    fake_count = 0

    for idx, row in df.iterrows():
        raw_label = row["label"]  # in this dataset: 1=REAL, 0=FAKE
        img_dict = row["image"]

        # Map to KryptosID schema: 0=Genuine, 1=Manipulated
        if raw_label == 1 and real_count < num_real:
            target_label = 0
            sub_folder = "real"
            gen_name = "pristine_photo"
            file_name = f"photo_real_{real_count:04d}.jpg"
            ident_id = f"photo_sub_real_{real_count:04d}"
            real_count += 1
        elif raw_label == 0 and fake_count < num_fake:
            target_label = 1
            sub_folder = "fake"
            gen_name = "ai_generated_portrait"
            file_name = f"photo_fake_{fake_count:04d}.jpg"
            ident_id = f"photo_sub_fake_{fake_count:04d}"
            fake_count += 1
        else:
            if real_count >= num_real and fake_count >= num_fake:
                break
            continue

        # Decode image bytes
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_resized = pil_img.resize((224, 224), Image.Resampling.LANCZOS)

        save_path = os.path.join(OUTPUT_DIR, sub_folder, file_name)
        img_resized.save(save_path, format="JPEG", quality=95)

        samples.append(DatasetSample(
            sample_id=f"{sub_folder}_{file_name[:-4]}",
            file_path=os.path.abspath(save_path),
            label=target_label,
            identity_id=ident_id,
            generator_name=gen_name,
            source_dataset="face-real-vs-fake",
        ))

    print(f"Extracted {real_count} real portrait photos and {fake_count} AI fake portrait photos.")
    return samples


def combine_with_video_data(
    photo_samples: List[DatasetSample],
    ff_manifest_path: str = "data/faceforensics_manifest.json",
    combined_out_path: str = "data/combined_manifest.json",
) -> SplitManifest:
    """Merges photo samples with existing FaceForensics++ video frames."""
    from training.splits import verify_split_manifest

    all_samples = list(photo_samples)

    if os.path.exists(ff_manifest_path):
        ff_manifest = verify_split_manifest(ff_manifest_path)
        for s in ff_manifest.train + ff_manifest.val + ff_manifest.test:
            all_samples.append(DatasetSample(
                sample_id=s["sample_id"],
                file_path=s["file_path"],
                label=s["label"],
                identity_id=s.get("identity_id") or f"id_{s['sample_id']}",
                generator_name=s.get("generator_name") or "ff_deepfake",
                source_dataset="FaceForensics++",
            ))
        print(f"Merged with {len(ff_manifest.train + ff_manifest.val + ff_manifest.test)} FaceForensics++ video frames.")

    manifest = create_identity_disjoint_split(
        samples=all_samples,
        dataset_name="KryptosID_MultiDomain_Photo_Video",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    manifest.save(combined_out_path)
    print(f"\nCombined Multi-Domain Manifest saved to {combined_out_path}:")
    print(f"  Train: {len(manifest.train)} samples")
    print(f"  Val:   {len(manifest.val)} samples")
    print(f"  Test:  {len(manifest.test)} samples")
    print(f"  SHA-256 Seal: {manifest.manifest_sha256}")
    return manifest


if __name__ == "__main__":
    photo_s = extract_photo_dataset(num_real=120, num_fake=120)
    combine_with_video_data(photo_s)
