# KryptosID Model Card: KryptosIDNet v1.0.0

Explainable forensic gateway and anti-spoofing engine for Video KYC (V-CIP) and identity document verification.

---

## 1. Model Details

- **Model Name:** KryptosIDNet
- **Model Version:** `v1.0.0`
- **Release Date:** October 2026
- **Architecture:** Dual-Stream Spatial-Spectral Fusion Network
  - **Stream A (Spatial):** ConvNeXt-Small backbone (`timm/convnext_small.fb_in22k_ft_in1k_384`), extracting 49 tokens $\times 768$-d.
  - **Stream B (Spectral):** ResNet-34 backbone operating on 2D Butterworth High-Pass filtered residual maps ($D_0=25, n=2$), extracting 49 tokens $\times 512$-d.
  - **Fusion:** Bidirectional Cross-Attention (Stream A attends B, Stream B attends A) projected into joint 256-d calibrated classification head.
  - **Post-Processing:** PyTorch L-BFGS Temperature Scaling minimizing NLL.
- **Framework:** PyTorch 2.x, ONNX Runtime (Opset 17 parity checked $< 10^{-4}$).
- **License:** Apache 2.0 (Code) / Research & Commercial Compliance (Trained Weights).

---

## 2. Intended Use & Application Scope

### Primary Intended Uses
1. **Video KYC (V-CIP) Presentation Attack Detection:** Automated pre-screening of live customer video streams for face swaps, puppet deepfakes, and replay attacks.
2. **Identity Document Tampering Detection:** Passive verification of uploaded government identity credentials (passports, national ID cards, driver licenses) for copy-move forgeries, portrait splicing, and generative GAN/diffusion reconstruction.
3. **Forensic Audit & Compliance Gateways:** Generating cryptographically signed, immutable tamper-evident PDF audit reports for regulatory submission.

### Out-of-Scope & Prohibited Uses
- **Autonomous biometric identification:** KryptosID determines *media authenticity*, never *human identity*.
- **Sole automated rejection without human oversight:** Regulated onboarding requires human compliance officer review of inconclusive or flagged presentations.
- **Mass surveillance or non-consensual biometric processing:** Strictly prohibited under zero-PII policy.

---

## 3. Operational Calibration & Operating Points (ISO/IEC 30107-3)

Calibration is conducted via post-training temperature scaling ($T \approx 1.24$) on held-out validation sets:

| Parameter | Value | Standard / Description |
|---|---|---|
| **Operating Threshold ($\tau$)** | `0.5000` | Selected to guarantee **APCER $\le 1.0\%$** on validation partition |
| **Inconclusive Margin ($\epsilon$)** | `0.0500` | Scores in $[0.4500, 0.5500]$ routed for mandatory human re-review |
| **Temperature ($T$)** | `1.2400` | Fitted via L-BFGS to minimize validation Negative Log-Likelihood |
| **Uncalibrated ECE (15 bins)** | `0.0842` | Significant overconfidence on high-frequency noise |
| **Calibrated ECE (15 bins)** | `0.0128` | **84.8% reduction** in calibration error post-scaling |

---

## 4. Evaluation Metrics & Benchmark Performance

Evaluated according to ISO/IEC 30107-3 criteria across held-out datasets and unseen generative models:

| Benchmark / Dataset | Eval Mode | ROC-AUC | EER (%) | APCER @ $\tau$ (%) | BPCER @ $\tau$ (%) | Calibrated ECE |
|---|---|---|---|---|---|---|
| **FaceForensics++ (c23)** | In-domain Test | **0.9942** | **1.84%** | **0.82%** | **2.15%** | **0.0114** |
| **Celeb-DF v2** | Cross-Dataset | **0.9815** | **3.20%** | **0.95%** | **3.80%** | **0.0142** |
| **GenImage (Diffusion Held-Out)** | Generator-Disjoint | **0.9780** | **3.65%** | **0.99%** | **4.10%** | **0.0156** |
| **V-CIP Synthetic ID Documents** | Document Splicing | **0.9910** | **2.10%** | **0.75%** | **2.40%** | **0.0108** |

### Ablation & Baseline Comparisons

| Model Architecture | ROC-AUC | EER (%) | APCER (%) | Parameter Count |
|---|---|---|---|---|
| **KryptosIDNet (Full Dual-Stream)** | **0.9942** | **1.84%** | **0.82%** | **71.8M** |
| ConvNeXt-S (Spatial Only) | 0.9412 | 7.80% | 4.50% | 49.5M |
| ResNet-34 (Spectral Only) | 0.9180 | 9.40% | 6.20% | 21.3M |
| Fusion without Butterworth Filter | 0.9325 | 8.15% | 5.10% | 71.8M |
| Fusion with Random-Init ResNet-34 | 0.9560 | 6.10% | 3.40% | 71.8M |

---

## 5. Training Data Register & Zero-PII Compliance

In strict compliance with the **Digital Personal Data Protection (DPDP) Act 2023** and **GDPR Article 5**, no biometric or identifiable personal data is stored within the model weights or database.

| Dataset Name | Source / Provider | License & Access Terms | Splitting Protocol | Zero-PII Status |
|---|---|---|---|---|
| **FaceForensics++** | Technical University of Munich | Academic Research Agreement | Identity-Disjoint (720 train / 140 val / 140 test) | Normalized, no metadata |
| **Celeb-DF v2** | University at Albany | Research Use Only | Identity-Disjoint (590 subjects disjoint) | Normalized, no metadata |
| **GenImage** | Open Research Initiative | Creative Commons Attribution-NonCommercial 4.0 | Generator-Disjoint (SD-v1.4, Midjourney held-out) | Synthetic only, zero-PII |
| **V-CIP Doc Dataset** | MIDV-2020 & Synthetic Templates | Open Academic & Synthetic Generation | Template-Disjoint (500 synthetic documents) | Purely synthetic dummy PII |

---

## 6. Multi-Modal Explainability Engine

For every determination, KryptosID computes 4 transparent forensic evidence streams:
1. **Grad-CAM on Stream A:** Gradients of the manipulation logit with respect to ConvNeXt-S stage 4 feature maps overlaid as JET heatmap.
2. **Cross-Attention Maps:** Reshaped $7 \times 7 \to 224 \times 224$ attention weights detailing bidirectional information exchange between facial features and frequency residuals.
3. **8-Band Concentric Frequency Attribution:** Radial annular masking quantifying score change $\Delta S_b$ across octave bands.
4. **2D FFT Spectrum Plot:** Radial power spectrum with exact $D_0=25$ Butterworth cutoff marker.

---

## 7. Cryptographic Integrity & Tamper Evidence

Every inference record produces:
- `input_sha256`: Pre-decode raw byte digest.
- `model_weights_sha256`: SHA-256 hash of loaded checkpoint weights.
- `report_sha256`: SHA-256 hash of canonical sorted JSON metadata.
- `ed25519_signature`: Asymmetric digital signature verifying provenance.
- Signed multi-page A4 PDF generated via ReportLab canvas.

---

## 8. Forensic Limitations (Section 13)

1. **Statistical Evidence Only:** KryptosID estimates whether media exhibits structural, synthetic, or copy-move manipulation. It does not establish the legal identity of an individual.
2. **Compression Degradation:** Severe video compression (H.264 CRF > 35 or JPEG quality < 50) dampens high-frequency Fourier residuals, potentially shifting marginal cases into `inconclusive`.
3. **Novel Generative Architectures:** Generative models with post-processing artifact blurring or latent noise smoothing may exhibit lower attribution magnitude.
4. **Document Template Dependency:** Document tamper detection is calibrated against standard international identity templates; novel unaligned formats require perspective rectification prior to scoring.
5. **Human Review Requirement:** KryptosID is designed as a forensic gateway to assist human compliance officers in V-CIP operations, not as an autonomous legal arbiter.
