"""Model components."""

import importlib

_LAZY = {
    "MelEncoder": "src.models.mel_encoder",
    "MambaContextTower": "src.models.mamba_tower",
    "CompressorParameterHead": "src.models.parameter_head",
    "DifferentiableDSPCompressor": "src.models.dsp_layer",
    "ContextAwareCompressor": "src.models.context_compressor",
}


def __getattr__(name):
    # Lazy imports so that importing one module (e.g. the DSP layer on a CPU-only
    # machine) doesn't pull in mamba-ssm, which needs CUDA.
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY)
