"""Central hardware selection for the CLI and Python inference API."""

import importlib
import os

import torch


_BACKENDS = {
    "auto": "auto", "nvidia": "cuda", "cuda": "cuda",
    "ascend": "npu", "910b": "npu", "npu": "npu",
    "cpu": "cpu", "apple": "mps", "mps": "mps",
}


def _npu_available(required=False):
    try:
        importlib.import_module("torch_npu")
    except ImportError as exc:
        if required:
            raise RuntimeError(
                "Ascend backend requires torch_npu. Use the 910B Docker image "
                "or install a torch_npu build matching PyTorch."
            ) from exc
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device(device=None):
    """Resolve --device/API overrides, otherwise VLX_SEEK_BACKEND (auto).

    Auto prefers CUDA, then Ascend, then CPU. Apple/MPS is reserved until its
    model and operator compatibility has been validated.
    """
    if device is None:
        backend = os.environ.get("VLX_SEEK_BACKEND", "auto").strip().lower()
        if backend not in _BACKENDS:
            raise ValueError(
                "VLX_SEEK_BACKEND must be auto, nvidia, ascend, cpu, or apple "
                "(reserved)."
            )
        device = _BACKENDS[backend]
    else:
        device = str(device).strip().lower()
        device = _BACKENDS.get(device, device)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif _npu_available():
            device = "npu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    kind = device.split(":", 1)[0]
    if kind == "mps":
        raise NotImplementedError(
            "Apple M-series/MPS is reserved but not yet supported by the "
            "VLX-Seek inference pipeline. Use a supported backend for now."
        )
    if kind == "npu":
        if not _npu_available(required=True):
            raise RuntimeError("Ascend backend selected, but no NPU is available.")
    elif kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA backend selected, but CUDA is unavailable.")
    elif kind != "cpu":
        raise ValueError(f"Unsupported inference device: {device}")
    return torch.device(device)


def attention_implementation(device):
    """Keep accelerated attention on CUDA/NPU and a portable CPU path."""
    return "flash_attention_2" if device.type in ("cuda", "npu") else "sdpa"
