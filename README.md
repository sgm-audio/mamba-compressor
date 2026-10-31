# MambaCompressor (research prototype)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch research prototype of a "context-aware" compressor: a stack of Mamba
SSM layers reads a log-mel spectrogram and predicts per-frame compressor
parameters, which drive a differentiable compressor applied to the audio.

**Status (2026-10-31):** model, loss and training-script code. No trained
checkpoint, no `from_pretrained`, no benchmark data. The differentiable
compressor (`dsp_layer.py`) has 7 CPU tests (`tests/test_dsp_layer.py`, all
passing with torch 2.15 CPU on Linux). The full model has not been run in this
repo's tests because `mamba-ssm` needs a CUDA GPU.

## What is in the code

| Part | File | What it does |
|---|---|---|
| Mel encoder | `src/models/mel_encoder.py` | `Linear(n_mels → d_model)` + `LayerNorm` |
| Context tower | `src/models/mamba_tower.py` | `mamba_ssm.Mamba` layers; defaults `n_layers=6`, `d_state=32`, `d_conv=4` |
| Parameter head | `src/models/parameter_head.py` | sigmoid-bounded `gain_db` [-40, 0] dB, `attack_ms` [0.1, 100], `release_ms` [10, 2000] |
| Differentiable compressor | `src/models/dsp_layer.py` | IIR attack/release smoothing; gain lookahead of `lookahead_frames` × hop (default 4 × 512 = 2048 samples, about 46 ms at 44.1 kHz). Offline only: the gain is applied early instead of delaying the audio. |
| Full model | `src/models/context_compressor.py` | `ContextAwareCompressor(audio, mel, return_params=False)`; build with `ContextAwareCompressor.from_config(cfg)` |
| Losses | `src/losses/` | multi-resolution STFT, LUFS target, crest factor |
| Data | `src/data/` | blind (synthetic corruption) and paired datasets |
| Training | `scripts/train_blind.py` | PyTorch Lightning blind pre-training |

## Usage

```python
import torch
from src.models.context_compressor import ContextAwareCompressor  # needs mamba-ssm + CUDA

model = ContextAwareCompressor(...)              # or ContextAwareCompressor.from_config(cfg)
out = model(audio, mel)                          # audio (B, T), log-mel (B, n_mels, frames)
out, params = model(audio, mel, return_params=True)
```

The DSP layer alone runs on CPU:

```python
from src.models.dsp_layer import DifferentiableDSPCompressor
comp = DifferentiableDSPCompressor(sample_rate=44100, hop_length=512, lookahead_frames=4)
y = comp(audio, {"gain_db": g, "attack_ms": a, "release_ms": r})   # each (B, frames)
```

## Tests

```bash
pip install torch pytest ruff     # CPU torch is enough
bash scripts/ci.sh                # compileall, ruff (errors only), pytest
```

## Install

```bash
git clone https://github.com/sgm-audio/mamba-compressor.git
cd mamba-compressor
pip install -r requirements.txt   # needs a CUDA build of PyTorch for mamba-ssm / causal-conv1d
```

## Train (blind pre-training)

```bash
# audio folder is set in configs/blind_pretrain.yaml (data.audio_dir)
python scripts/train_blind.py --config configs/blind_pretrain.yaml --gpus 0
```

Flags that exist: `--config`, `--gpus`, `--epochs`, `--batch_size`, `--resume`.
Fine-tuning, evaluation and ONNX/TensorRT export scripts are not written yet.

## Not implemented yet

- Pretrained weights / `from_pretrained`
- `scripts/finetune_paired.py`, `evaluate.py`, `export_onnx.py`, `export_trtllm.py`
- Streaming inference. `MambaContextTower.forward_with_state()` raises
  `NotImplementedError`; it needs `Mamba.step()` and a GPU to verify.
- Tests for the Mamba tower and the full model (they need a CUDA box)
- A ballistics kernel faster than the per-sample Python loop in `dsp_layer.py`
  (about 15 µs/sample on a loaded 8-thread Xeon, CPU torch; fine for training
  experiments, not for real time)
- Latency or memory benchmarks (none have been measured)
- NVIDIA NeMo integration (not imported anywhere; commented out in `requirements.txt`)

## Fixed on 2026-10-31

- Lookahead ran backwards: the gain envelope was shifted later instead of
  earlier, so a gain drop landed about 2077 samples (47 ms) late. The step test
  in `tests/test_dsp_layer.py` covers it.

## Licence

MIT — see [LICENSE](LICENSE). Mamba SSM by Albert Gu and Tri Dao
([state-spaces/mamba](https://github.com/state-spaces/mamba)).

Contact: scott@sgmstudios.ca
