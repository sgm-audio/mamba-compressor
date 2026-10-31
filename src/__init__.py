"""MambaCompressor: research code for a Mamba-driven differentiable compressor."""

__version__ = "0.1.0"
__author__ = "Scott Mills"

import importlib

_LAZY = {
    "ContextAwareCompressor": "src.models.context_compressor",
    "MambaContextTower": "src.models.mamba_tower",
    "DifferentiableDSPCompressor": "src.models.dsp_layer",
    "MelEncoder": "src.models.mel_encoder",
    "MultiResolutionSTFTLoss": "src.losses.spectral_loss",
    "CrestFactorLoss": "src.losses.crest_factor",
    "LUFSLoss": "src.losses.lufs_loss",
    "SyntheticCorruptor": "src.data.blind_dataset",
    "BlindAudioDataset": "src.data.blind_dataset",
    "load_audio": "src.utils.audio",
    "save_audio": "src.utils.audio",
    "mel_spectrogram": "src.utils.audio",
    "compute_lufs": "src.utils.metrics",
    "compute_crest_factor": "src.utils.metrics",
    "compute_spectral_flatness": "src.utils.metrics",
}


def __getattr__(name):
    # Lazy imports so that importing one module (e.g. the DSP layer on a CPU-only
    # machine) doesn't pull in mamba-ssm, which needs CUDA.
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY)
