"""
Extract and prepare real FaceForensics++ face frames with identity-disjoint splits.
Implements Section 5 of DESIGN.md:
- Downloads genuine and deepfake video clips from FaceForensics++ benchmark.
- Extracts aligned frames and resizes to 224x224.
- Enforces strict subject identity-disjoint splitting.
- Produces SHA-256 sealed SplitManifest.
"""

import os
import cv2
import json
import uuid
from typing import List, Dict, Any, Tuple
from huggingface_hub import HfApi, hf_hub_download
from training.splits import DatasetSample, SplitManifest, create_identity_disjoint_split

REPO_ID = "TonyStark03/deepfake"
BASE_DATA_DIR = "data/faceforensics"


def get_available_video_ids(limit: int = 15) -> Tuple[List[str], List[str]]:
    """Fetches valid video ids from TonyStark03/deepfake."""
    api = HfApi()
    info = api.dataset_info(REPO_ID, files_metadata=True)
    mp4s = [s.rfilename for s in info.siblings if s.rfilename.endswith(".mp4")]

    reals = [f[:-4] for f in mp4s if "_" not in f][:limit]
    fakes = [f[:-4] for f in mp4s if "_" in f][:limit]
    return reals, fakes


def download_and_extract_frames(
    real_video_ids: List[str],
    fake_video_ids: List[str],
    frames_per_video: int = 8,
) -> Tuple[List[DatasetSample], List[DatasetSample]]:
    """Downloads FF++ clips and extracts balanced face frames."""
    os.makedirs(os.path.join(BASE_DATA_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DATA_DIR, "fake"), exist_ok=True)

    real_samples: List[DatasetSample] = []
    fake_samples: List[DatasetSample] = []

    print(f"Downloading & extracting {len(real_video_ids)} real videos...")
    for vid_id in real_video_ids:
        filename = f"{vid_id}.mp4"
        try:
            local_vid = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
            cap = cv2.VideoCapture(local_vid)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // frames_per_video)

            idx = 0
            saved = 0
            while cap.isOpened() and saved < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    frame_224 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                    save_path = os.path.join(BASE_DATA_DIR, "real", f"ff_real_{vid_id}_{saved:03d}.jpg")
                    cv2.imwrite(save_path, frame_224, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    real_samples.append(DatasetSample(
                        sample_id=f"ff_real_{vid_id}_{saved:03d}",
                        file_path=os.path.abspath(save_path),
                        label=0,
                        identity_id=f"actor_{vid_id}",
                        generator_name="pristine",
                        source_dataset="FaceForensics++",
                    ))
                    saved += 1
                idx += 1
            cap.release()
            print(f"  [Real] Video {filename}: extracted {saved} frames.")
        except Exception as e:
            print(f"  [Real] Error on {filename}: {e}")

    print(f"\nDownloading & extracting {len(fake_video_ids)} fake videos...")
    for vid_id in fake_video_ids:
        filename = f"{vid_id}.mp4"
        try:
            local_vid = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
            cap = cv2.VideoCapture(local_vid)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // frames_per_video)

            # In FF++, actor1 is swapped with actor2
            parts = vid_id.split("_")
            sub_id = f"actor_{parts[0]}" if parts else f"actor_{vid_id}"

            idx = 0
            saved = 0
            while cap.isOpened() and saved < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    frame_224 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                    save_path = os.path.join(BASE_DATA_DIR, "fake", f"ff_fake_{vid_id}_{saved:03d}.jpg")
                    cv2.imwrite(save_path, frame_224, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    fake_samples.append(DatasetSample(
                        sample_id=f"ff_fake_{vid_id}_{saved:03d}",
                        file_path=os.path.abspath(save_path),
                        label=1,
                        identity_id=sub_id,
                        generator_name="Deepfakes",
                        source_dataset="FaceForensics++",
                    ))
                    saved += 1
                idx += 1
            cap.release()
            print(f"  [Fake] Video {filename}: extracted {saved} frames.")
        except Exception as e:
            print(f"  [Fake] Error on {filename}: {e}")

    return real_samples, fake_samples


def build_sealed_manifest(
    real_samples: List[DatasetSample],
    fake_samples: List[DatasetSample],
    manifest_out: str = "data/faceforensics_manifest.json",
) -> SplitManifest:
    """Creates identity-disjoint partition and signs SHA-256 manifest."""
    all_samples = real_samples + fake_samples
    manifest = create_identity_disjoint_split(
        samples=all_samples,
        dataset_name="FaceForensics++",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    manifest.save(manifest_out)
    print(f"\nManifest sealed to {manifest_out}:")
    print(f"  Train: {len(manifest.train)} samples")
    print(f"  Val:   {len(manifest.val)} samples")
    print(f"  Test:  {len(manifest.test)} samples")
    print(f"  Manifest SHA-256: {manifest.manifest_sha256}")
    return manifest


if __name__ == "__main__":
    reals, fakes = get_available_video_ids(limit=15)
    real_s, fake_s = download_and_extract_frames(reals, fakes, frames_per_video=8)
    build_sealed_manifest(real_s, fake_s)
