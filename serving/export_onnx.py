"""ONNX Export and PyTorch/ONNX-Runtime Parity Verification.

Implements Section 10.2 of DESIGN.md:
- Exports KryptosIDNet to ONNX format (opset 17).
- Preserves Butterworth high-pass filter as a static registered tensor buffer.
- Parity check: Verifies maximum absolute difference between PyTorch and ONNX Runtime logits is < 1e-4.
"""

from typing import Tuple
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from core.model import KryptosIDNet


class ExportableKryptosIDNet(nn.Module):
    """Wrapper ensuring dual-tensor or tuple inputs export cleanly to ONNX."""

    def __init__(self, base_model: KryptosIDNet) -> None:
        super().__init__()
        self.model = base_model

    def forward(self, spatial: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        return self.model((spatial, spectral))


def export_to_onnx(
    model: KryptosIDNet,
    output_path: str = "kryptosid.onnx",
    opset_version: int = 17,
    tolerance: float = 1e-4,
) -> Tuple[str, float]:
    """Export KryptosIDNet to ONNX (opset 17) and strictly enforce parity check (< 1e-4).
    
    Args:
        model: Trained KryptosIDNet PyTorch model.
        output_path: Destination ONNX model file.
        opset_version: Target ONNX opset (default 17 per Section 10.2).
        tolerance: Maximum allowable absolute difference between PyTorch and ONNX logits.
        
    Returns:
        Tuple of (output_path, max_absolute_difference).
        
    Raises:
        AssertionError: If max absolute parity difference >= tolerance.
    """
    model.eval()
    exportable = ExportableKryptosIDNet(model)
    exportable.eval()

    dummy_spatial = torch.randn(1, 3, 224, 224)
    dummy_spectral = torch.randn(1, 1, 224, 224)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True) if os.path.dirname(output_path) else None

    # Export to ONNX
    torch.onnx.export(
        exportable,
        (dummy_spatial, dummy_spectral),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["spatial_input", "spectral_input"],
        output_names=["logits"],
        dynamic_axes={
            "spatial_input": {0: "batch_size"},
            "spectral_input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Verify ONNX model definition
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Parity check with ONNX Runtime
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    # Test on multiple deterministic test inputs
    test_spatial = torch.randn(2, 3, 224, 224)
    test_spectral = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        pytorch_logits = exportable(test_spatial, test_spectral).cpu().numpy()

    ort_inputs = {
        "spatial_input": test_spatial.numpy(),
        "spectral_input": test_spectral.numpy(),
    }
    ort_logits = session.run(["logits"], ort_inputs)[0]

    max_diff = float(np.max(np.abs(pytorch_logits - ort_logits)))

    if max_diff >= tolerance:
        raise AssertionError(
            f"ONNX parity check failed! Max difference {max_diff:.6e} exceeds tolerance {tolerance:.6e}"
        )

    return output_path, max_diff
