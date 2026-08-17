"""Device selection and reporting.

The benchmark hardware is an NVIDIA H100, but everything here degrades
gracefully to CPU / any CUDA GPU so the repo remains runnable on a
laptop for development and on the evaluator's machine for scoring.
"""
from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device.

    Args:
        prefer: "auto" | "cuda" | "cpu". "auto" picks CUDA if available.
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    if torch.cuda.is_available() and prefer in ("auto", "cuda"):
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict:
    """Small dict of device info worth logging alongside every run/result,
    since runtime numbers are meaningless without the hardware context."""
    info = {"device": str(device)}
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(idx),
                "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(idx)),
                "total_memory_gb": round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 2),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
        )
    else:
        info.update({"torch_version": torch.__version__})
    return info
