# KryptosID: Design Document

Explainable forensic gateway and anti-spoofing engine for Video KYC (V-CIP) and identity document verification.

This document describes how each component is built and how the components connect. It also records the corrections from the earlier design review, so the implementation starts from the corrected version.

---

## 1. System Overview

KryptosID takes an image, a video, or an identity document as input and returns:

1. A **decision**: `genuine`, `manipulated`, or `inconclusive`, based on a calibrated threshold.
2. A **score** with calibrated probability and the operating point used.
3. An **explanation**: Grad-CAM heatmaps, cross-attention maps, and frequency-band attribution.
4. A **signed forensic report** (PDF) with hashes, model identity, and timestamps.

```
            +-----------------------------------------------------------+
            |                       Streamlit UI                        |
            |     (upload, results, explanations, report download)      |
            +-------------------------------+---------------------------+
                                            | HTTPS (JSON / multipart)
            +-------------------------------v---------------------------+
            |                   FastAPI Inference Service                |
            |  /v1/analyze   /v1/reports/{id}   /v1/health   /v1/model  |
            +----+------------------+------------------+----------------+
                 |                  |                  |
     +-----------v-----+  +---------v--------+  +------v---------------+
     | Ingest &        |  | KryptosIDNet     |  | Forensics            |
     | Preprocess      |  | (core/model.py)  |  | (hash, report, sign) |
     | (decode, face/  |->| Stream A + B +   |->| ReportLab PDF,       |
     |  doc crop, JPEG |  | Cross-Attention  |  | Ed25519 signature    |
     |  Q=85 normalize)|  | + calibrated head|  +------+---------------+
     +-----------------+  +---------+--------+         |
                                    |                  v
                          +---------v--------+   +----------------+
                          | Explainability   |   | Report store   |
                          | Grad-CAM, attn   |   | (SQLite/PG +   |
                          | maps, band attr. |   |  object store) |
                          +------------------+   +----------------+
```

The **liveness check** (blink or challenge-response) runs as a separate module in the video path. Its result is fused with the artifact score in the decision layer (Section 8).

---

## 2. Repository Layout

```
kryptosid/
├── configs/
│   ├── train_base.yaml          # hyperparameters, seeds, data paths
│   ├── eval_crossdataset.yaml   # evaluation protocol
│   └── serve.yaml               # thresholds, model path, limits
├── core/
│   ├── model.py                 # KryptosIDNet (both streams + fusion + head)
│   ├── frequency.py             # Butterworth high-pass via FFT
│   ├── fusion.py                # CrossAttentionFusion (bidirectional)
│   ├── preprocess.py            # decode, resize, JPEG Q=85 normalize, ImageNet norm
│   ├── symmetry.py              # D4 group actions for copy-move checks
│   ├── calibrate.py             # temperature scaling, ECE, threshold selection
│   └── explain.py               # Grad-CAM, attention maps, band attribution
├── training/
│   ├── datasets.py              # FF++, Celeb-DF, GenImage, doc datasets
│   ├── splits.py                # identity-disjoint and generator-disjoint splits
│   ├── augment.py               # JPEG/H.264 re-encoding augmentations
│   ├── train.py                 # training loop (AMP, checkpointing, seeds)
│   └── evaluate.py              # AUC, EER, APCER, BPCER, ECE, per-generator tables
├── video/
│   ├── sampler.py               # frame sampling, face detection and alignment
│   ├── liveness.py              # blink / challenge-response check
│   └── aggregate.py             # temporal aggregation of per-frame scores
├── document/
│   ├── locate.py                # document detection and perspective rectification
│   └── regions.py               # photo and MRZ region extraction
├── forensics/
│   ├── hashing.py               # SHA-256 of inputs, weights, configs
│   ├── report.py                # ReportLab canvas PDF generation
│   ├── signing.py               # Ed25519 signing and verification
│   └── store.py                 # report persistence
├── serving/
│   ├── api.py                   # FastAPI app
│   ├── schemas.py               # Pydantic request/response models
│   ├── inference.py             # model loading, batching, decision logic
│   └── export_onnx.py           # ONNX export and parity check
├── app/
│   ├── streamlit_app.py         # UI entry point
│   ├── components.py            # result cards, heatmap viewer, copy button
│   └── theme.py                 # CSS tokens, dark mode, print styles
├── tests/                       # unit, property, and regression tests
├── docker/
│   └── Dockerfile
├── MODEL_CARD.md                # intended use, limits, metrics, dataset licenses
├── DESIGN.md                    # this file
└── README.md
```

---

## 3. Input Pipeline

### 3.1 Ingest
- Accepted types: JPEG, PNG, WebP for images. MP4 or WebM for video. PDF or JPEG scans for documents.
- The raw bytes are hashed with SHA-256 **before any decoding**. This hash is the evidence identifier and is written into the report.
- Limits: maximum file size, maximum video duration, and maximum frame count are set in `serve.yaml`. Requests above the limits are rejected with HTTP 413.

### 3.2 Normalization (training and inference must match)
1. Decode to RGB (uint8).
2. **JPEG Q=85 normalization**: re-encode the image at quality 85 with OpenCV. This removes the compression shortcut, so the network cannot read the original quality factor as a label signal.
3. Resize to 224 × 224 (aspect-preserving crop around the detected face or document region).
4. Scale to float in [0, 1], then apply ImageNet normalization:
   - mean `[0.485, 0.456, 0.406]`
   - std `[0.229, 0.224, 0.225]`

### 3.3 Stream B input (spectral)
- Convert the normalized RGB tensor to grayscale: `Y = 0.299 R + 0.587 G + 0.114 B`, computed on the un-normalized [0, 1] values before ImageNet normalization.
- Pass it to the Butterworth high-pass filter (Section 4.2) to produce the residual `[B, 1, 224, 224]`.

### 3.4 Video path
- Sample frames at a fixed rate (for example, 4 fps).
- Detect and align faces. Each frame becomes a 224 × 224 crop.
- Score each frame with KryptosIDNet, then aggregate over time (Section 8).

### 3.5 Document path
- Detect the document quadrilateral, rectify it to a canonical size, and extract the photo region and the MRZ region.
- Run the artifact detector on the photo region and on the full rectified page.
- Copy-move checks use `core/symmetry.py` on the full page (Section 4.4).

---

## 4. Model: KryptosIDNet

Implemented in `core/model.py`. Components are described in the order data flows through them.

### 4.1 Stream A: Spatial (ConvNeXt-Small)

- Backbone: `timm.create_model('convnext_small', pretrained=True, num_classes=0)`.
- **Important:** with `num_classes=0`, timm returns pooled vectors, which discard the spatial layout that cross-attention needs. Use `forward_features()` instead.
- Input: `[B, 3, 224, 224]`.
- Output: feature map `[B, 768, 7, 7]`, flattened to **49 spatial tokens** of dimension 768: `[B, 49, 768]`.
- A linear projection maps tokens to 256: `[B, 49, 256]`.

### 4.2 Stream B: Spectral (Butterworth high-pass → ResNet-34)

**Butterworth high-pass filter** (`core/frequency.py`):

- Grid: 224 × 224. Frequency coordinates are computed with `torch.fft.fftfreq` and shifted with `torch.fft.fftshift`, so the DC component sits at (112, 112).
- Distance from center in index units: `D(u, v) = sqrt((u - 112)² + (v - 112)²)`.
- Transfer function: `H(u, v) = 1 / (1 + (D0 / D(u, v))^(2n))`, with `D0 = 25` and `n = 2`. At `D = 0` the response is set to 0 to avoid division by zero.
- With a 224-pixel grid, the Nyquist frequency corresponds to index 112. So `D0 = 25` removes content below roughly 22% of Nyquist. The filter is stored as a non-trainable buffer.
- Filtering:
  1. `F = fftshift(fft2(x))`
  2. `F_hp = F * H`
  3. `x_hp = real(ifft2(ifftshift(F_hp)))`
  4. Per-sample standardization (zero mean, unit variance), because the residual's scale is very different from ImageNet inputs.

**Backbone:** `timm.create_model('resnet34', in_chans=1, pretrained=True, num_classes=0)`, again using `forward_features()`.
- Input: `[B, 1, 224, 224]`.
- Output: feature map `[B, 512, 7, 7]`, flattened to **49 spectral tokens** of dimension 512: `[B, 49, 512]`.
- A linear projection maps tokens to 256: `[B, 49, 256]`.

**Pretraining ablation:** train one variant with ImageNet-pretrained ResNet-34 and one with random initialization. Keep whichever generalizes better across datasets. Do not assume the pretrained weights help.

### 4.3 Cross-Attention Fusion (`core/fusion.py`)

This replaces the single-token design. Each stream attends to the other, so the attention runs over real spatial positions.

- Inputs: spatial tokens `S ∈ [B, 49, 256]`, spectral tokens `P ∈ [B, 49, 256]`.
- Block 1 (spatial queries spectral): `Q = S W_q`, `K = P W_k`, `V = P W_v`.
  `A_sp = softmax(Q Kᵀ / sqrt(256)) V`, shape `[B, 49, 256]`.
- Block 2 (spectral queries spatial): the same with roles swapped, `A_pe = softmax(P' Sᵀ / sqrt(256)) S`.
- Residual and LayerNorm on each block output.
- Pooling: mean over the 49 tokens of each attended stream, then sum, giving a fused vector `[B, 256]`.
- Attention weights are returned alongside the output for explainability.
- Single head is used to match the specification. Multi-head (for example, 4 heads) is a tested option if the single-head version underfits.

### 4.4 Classification Head

`LayerNorm(256) → Linear(256, 64) → GELU → Dropout(0.2) → Linear(64, 1)`

- Output: one logit per sample.
- Loss: `torch.nn.BCEWithLogitsLoss`.
- Probability: `sigmoid(logit / T)`, where `T` is the temperature learned on the validation set (Section 6.3).

### 4.5 Symmetry Check (`core/symmetry.py`)

This is a separate forensic check, not part of the learned network.

- Group: **D4** (the symmetry group of the square). It has 8 elements: four rotations (0°, 90°, 180°, 270°) and four reflections (horizontal, vertical, and the two diagonals).
- Use: copy-move detection on documents. Patches are matched against the image under all 8 D4 transforms. A high-similarity match between non-overlapping regions under a transform flags a candidate copy-move.
- Output: candidate region pairs, the transform used, and a similarity score. These appear in the report as evidence, not as a verdict.

---

## 5. Data

### 5.1 Datasets

| Role | Dataset | Notes |
|---|---|---|
| Real faces | Celeb-DF v2, FF++ pristine (1,000 sequences) | Check each license before use. |
| Face manipulation | FaceForensics++ (Deepfakes, FaceSwap, NeuralTextures, Face2Face) | Access request required. |
| Generated images | GenImage, plus curated diffusion sets | Verify which generators are included in the current release (do not assume FLUX.1 is present). |
| Document splicing | DocTamper, CASIA 2.0 | General splicing data, not ID documents. Use for pretraining only. |
| Identity documents | Licensed or synthetic ID templates | Required for V-CIP credibility. Real PII requires a documented legal basis under applicable law (for example, India's DPDP Act). |

A **dataset register** (`configs/datasets.yaml`) records for each dataset: source URL, version, license, commercial-use permission, access terms, and the date checked.

### 5.2 Splits

Splits are a leakage control and are defined in `training/splits.py`.

- **Face datasets:** split by identity (or by source actor pair for FF++). No identity appears in more than one split. Splitting by frame is not allowed.
- **Generated images:** there is no subject identity. Split by **generator** (held-out generators in test) and by **seed or source prompt group**. Report results per generator.
- **Cross-dataset:** train on FF++ and GenImage, test on Celeb-DF and held-out generators. This is the main generalization result.
- Split manifests are written to disk and hashed. Training refuses to start if the manifest hash changed without a version bump.

### 5.3 Augmentation (`training/augment.py`)

Applied on the fly during training:

- JPEG re-encoding at Q=85 for all inputs (normalization).
- Additional random JPEG quality in [60, 95] with probability 0.5, so the model does not depend on exactly Q=85.
- H.264 re-encoding at random CRF values for video frames (requires `ffmpeg`).
- Random resized crop (scale 0.85–1.0), horizontal flip, mild color jitter.
- Gaussian noise with small sigma (adversarial robustness test only, not training by default).

---

## 6. Training and Calibration

### 6.1 Training (`training/train.py`)

Two-stage schedule:

1. **Warm-up:** backbones frozen, only fusion and head train. Learning rate 1e-3, 3 epochs.
2. **Fine-tune:** unfreeze the last stages of both backbones. Backbone learning rate 1e-5, head 1e-4. AdamW, weight decay 0.05, cosine schedule with warm-up.

Other settings:

- Batch size 32 per GPU, mixed precision (AMP), gradient clipping at 1.0.
- Fixed seeds (Python, NumPy, PyTorch, CUDA). The seed and the full config are saved with every checkpoint.
- Early stopping on validation **EER**, not loss.
- Checkpoints store: weights, config hash, dataset manifest hash, git commit, and training metrics.

### 6.2 Baselines and Ablations

Each row is trained with the same splits and augmentations:

| Model | Purpose |
|---|---|
| ConvNeXt-S spatial only | Measures the value of the spectral stream. |
| ResNet-34 spectral only | Measures the value of the spatial stream. |
| Fusion without the Butterworth filter | Measures the value of the high-pass filter. |
| Fusion with random init ResNet-34 | Tests the pretraining claim. |
| XceptionNet (published detector) | External reference point. |
| CLIP linear probe | Strong modern reference. |

### 6.3 Calibration (`core/calibrate.py`)

- **Temperature scaling:** fit a single scalar `T` on the validation set by minimizing NLL. Model weights are not changed.
- **ECE** (expected calibration error, 15 bins) is reported before and after scaling.
- **Threshold selection:** choose the decision threshold that achieves a target **APCER** on validation (for example, APCER ≤ 1%). Record the resulting BPCER on the same set.
- The threshold is saved in `serve.yaml`. It is not chosen on the test set.

### 6.4 Metrics (`training/evaluate.py`)

Reported for each dataset and each held-out generator:

- **AUC** and **EER**.
- **APCER** (attack presentation classification error rate) and **BPCER** (bona fide presentation classification error rate), following ISO/IEC 30107-3, at the operating threshold.
- **ECE** after temperature scaling.
- Confusion matrices and failure-case galleries.

Results are written to `reports/eval_<date>.json` and the model card is generated from that file, so the numbers in documents always match the evaluation run.

---

## 7. Explainability (`core/explain.py`)

Each explanation is computed for every decision.

1. **Grad-CAM on Stream A:** gradients of the logit with respect to the last ConvNeXt stage. Shown as a heatmap over the input face or document crop.
2. **Cross-attention maps:** the attention weights from both fusion blocks, reshaped to 7 × 7 and upsampled to 224 × 224. These show which spatial regions of one stream drew on which regions of the other.
3. **Frequency-band attribution:** the spectrum is split into bands (for example, 8 concentric rings). Each band is removed in turn and the score change is recorded. The bar chart shows which frequency bands drove the decision.
4. **Spectrum plot:** log-magnitude spectrum of the input, with the Butterworth cutoff circle drawn.

Each explanation is a PNG saved with the report. The explanation is labeled as evidence for review, not as a proof of manipulation.

---

## 8. Video Decision Logic (`video/`)

1. Sample frames and detect faces (`sampler.py`).
2. Score each frame with KryptosIDNet. Scores are calibrated probabilities.
3. Aggregate (`aggregate.py`): use the mean of the top-k frame scores (k = 10% of frames) and a temporal consistency term that penalizes large score jumps between adjacent frames.
4. Liveness (`liveness.py`): blink or a random head-turn challenge. The session fails liveness if the challenge is not completed within the time window.
5. Final decision:
   - `manipulated` if the aggregated artifact score is above the high threshold, or liveness fails with a high artifact score.
   - `inconclusive` if the score is between thresholds, or if too few faces were detected.
   - `genuine` otherwise.
   - Every decision includes which rule produced it.

Thresholds come from calibration (Section 6.3), not from the UI.

---

## 9. Forensic Report (`forensics/`)

### 9.1 Evidence chain

For every analysis:

1. `input_sha256`: SHA-256 of the raw uploaded bytes.
2. `model`: model name, version, and `weights_sha256`.
3. `config_sha256`: hash of the config used at inference.
4. `threshold` and `temperature` used.
5. `started_at` and `finished_at`, in UTC ISO 8601.
6. `decision`, `score`, and `rule_fired`.
7. Explanation images, each with its own SHA-256.
8. `report_sha256`: hash of the canonical JSON of the fields above.

### 9.2 Signing

- The canonical JSON (sorted keys, no extra whitespace) is signed with **Ed25519** (`forensics/signing.py`).
- The signature and the public key identifier are embedded in the PDF and stored in the report store.
- A `verify` endpoint and CLI command recompute the hashes and check the signature. Any change to the file or the metadata breaks verification.
- Private keys live outside the repository (environment or a KMS). Key rotation is recorded with a key ID.
- Optional: RFC 3161 timestamp from a trusted timestamping authority for independent time evidence.

### 9.3 PDF generation (`forensics/report.py`)

- Built with `reportlab.pdfgen.canvas` on `A4`.
- Sections: summary (decision, score, threshold), evidence chain table, explanation figures, limitations statement, and verification instructions.
- The last page prints the verification command and the report ID so the report can be checked later.

### 9.4 Storage

- Metadata in SQLite for development, PostgreSQL for production.
- Raw inputs are kept only if the retention setting allows it. Otherwise only hashes and derived artifacts are stored.
- Report access requires authentication (out of scope for this document, but the endpoint is designed to sit behind it).

---

## 10. Serving

### 10.1 Inference service (`serving/`)

- **FastAPI** application (`api.py`), with Pydantic schemas (`schemas.py`).
- Endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/analyze` | Upload media, returns decision, scores, explanation IDs, report ID. |
| GET | `/v1/reports/{id}` | Report metadata and PDF download. |
| POST | `/v1/reports/{id}/verify` | Recomputes hashes and checks the signature. |
| GET | `/v1/health` | Liveness probe and model load status. |
| GET | `/v1/model` | Model card fields: version, weights hash, threshold, dataset list. |

- `inference.py` loads the model once at startup, runs batched inference, and enforces timeouts.
- Long videos are processed as background jobs. `/v1/analyze` returns a job ID, and the client polls for the result.

### 10.2 Model export

- `export_onnx.py` exports the trained model to ONNX (opset 17). The Butterworth filter is exported as a fixed buffer, so it is not recomputed from parameters at runtime.
- Parity check: maximum absolute difference between PyTorch and ONNX Runtime logits must be below 1e-4 on the test set. The export fails if the check fails.

### 10.3 Packaging

- Docker image based on a slim Python 3.10+ base, with CPU inference by default and a GPU variant.
- Config comes from environment variables and the mounted `configs/serve.yaml`.
- Health checks wired to `/v1/health`.

### 10.4 Operational limits

- Request size and duration limits enforced before decoding.
- Rate limiting per API key.
- Structured JSON logs. Logs include request IDs and report IDs but never raw media or PII.

---

## 11. Streamlit Front End (`app/`)

The UI calls the FastAPI service over HTTP. It holds no model code, so the UI and the inference service can be deployed and scaled separately.

### 11.1 Pages

1. **Analyze:** upload area (image, video, or document), an analyze button, and a progress indicator for video jobs.
2. **Result:** decision badge, score with threshold marker, explanation tabs (heatmap, attention, band attribution, spectrum), and report download.
3. **Verify:** upload a report PDF or enter a report ID to check the signature.
4. **About:** model card, limitations, and dataset licenses.

### 11.2 UI features from the preference list

Streamlit cannot run arbitrary JavaScript in the page, so some features need a custom component or a small static frontend. The table shows how each one is built.

| Feature | Implementation |
|---|---|
| Dark mode toggle | CSS variables in `theme.py`, switched by a `st.toggle` and a session value. |
| Simple cookie banner | Custom component, shown until consent is stored in a cookie. |
| Site search | `st.text_input` filtering the About page sections and report IDs. |
| Top button | Custom component with anchor link and fixed position CSS. |
| Mobile menu | Sidebar collapse with a visible toggle on narrow widths (CSS media query). |
| Loading animations | `st.spinner` for short work, custom CSS animation for video job progress. |
| Hover states | CSS `:hover` rules in `theme.py`. |
| Scroll progress bar | Custom component with a scroll listener, or omitted on Streamlit pages if the component cost is not justified. |
| Copy button | Custom component using the clipboard API, used for report IDs and hashes. |
| Print stylesheet | `@media print` rules in `theme.py` for the About and Result pages. |
| Sticky header | Fixed-position CSS header. |
| Skip to content | Visually hidden link at the top of the page, targeting the main region. |
| Password visibility toggle | Used only if a login form is added. |
| UTM tracking | Read UTM parameters from the URL and store them in session state, never with personal data. |
| Form success state | Message and status after a successful upload or verification. |
| Form error state | Field-level and form-level error messages with the error text from the API. |
| Confirmation modals | Used before deleting a report or sending a video for processing. |
| Last updated date | Footer showing the model card date and the app build date. |
| Floating contact | Fixed button linking to the support contact. |
| Expandable FAQ | `st.expander` on the About page. |

Accessibility requirements apply to all components: keyboard navigation, visible focus, contrast ratio of at least 4.5:1 in both themes, and text alternatives for every explanation image.

---

## 12. Testing

- **Unit tests:** Butterworth filter (response at DC, at D0, and at Nyquist), D4 transforms (closure and inverse), hashing, signing round-trip.
- **Shape tests:** every tensor boundary in `KryptosIDNet` (`[B, 49, 768]`, `[B, 49, 512]`, `[B, 256]`).
- **Regression tests:** a fixed set of images with expected scores, to catch silent changes after retraining.
- **Parity tests:** ONNX versus PyTorch logits.
- **Report tests:** generated PDF verifies, and any single-byte change fails verification.
- **Security tests:** oversized inputs, malformed images, and zip or video bombs are rejected.

CI runs unit, shape, report, and parity tests on every commit. Evaluation runs on a schedule and on each model release.

---

## 13. Limitations (to be stated in the model card and report)

- The model estimates whether media shows signs of manipulation or generation. It does not establish the identity of a person.
- Performance drops on compression, resolution, and generators outside the training set. Per-generator results are published.
- Adversarial attacks can reduce accuracy. Adversarial results are reported, not hidden.
- A decision is evidence for human review. It is not a final determination for regulated onboarding.
- Identity document checks depend on the document templates in the training data.

---

## 14. Build Order

1. `core/frequency.py`, `core/fusion.py`, `core/model.py`, with shape tests.
2. `training/splits.py` and dataset register. Confirm licenses before any download.
3. `training/train.py` with baselines (Section 6.2).
4. `core/calibrate.py` and `training/evaluate.py`. Produce the first results table.
5. `core/explain.py`.
6. `forensics/` (hashing, report, signing) and the verify path.
7. `serving/` (FastAPI, inference, ONNX export with parity check).
8. `app/` (Streamlit pages, then the UI features in Section 11.2).
9. Docker packaging, model card generation, and public results with failure cases.

Each step ends with tests passing and its outputs written to disk, so a later step can be rerun without redoing the earlier ones.

---

## 15. Changes from the Earlier Plan

| Earlier plan | Corrected design |
|---|---|
| Single-token cross-attention (softmax over one key is always 1). | Bidirectional attention over 49 spatial tokens per stream (Section 4.3). |
| "Dihedral D5". | D4 (square symmetry group) for copy-move checks (Section 4.5). |
| D₀ = 25 with no grid. | Defined on a 224 × 224 grid with DC at center and Nyquist at index 112 (Section 4.2). |
| Split by subject for all data. | Identity split for faces, generator split for generated images (Section 5.2). |
| Dataset list without licenses. | Dataset register with license and access terms (Section 5.1). |
| Document datasets for V-CIP. | Licensed or synthetic ID templates, with a PII basis documented (Section 5.1). |
| No calibration method. | Temperature scaling, ECE, and APCER-based threshold (Section 6.3). |
| No explainability method. | Grad-CAM, attention maps, and band attribution (Section 7). |
| Unsigned output. | Hashed evidence chain with Ed25519 signature and verify endpoint (Section 9). |
