# MambaCompressor (research prototype)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch research prototype of a "context-aware" compressor: a stack of Mamba
SSM layers reads a log-mel spectrogram and predicts per-frame compressor
parameters, which drive a differentiable compressor applied to the audio.

**Status (checked 2026-10-31):** model, loss and training-script code only.
There is no trained checkpoint, no `from_pretrained`, no test suite, and no
benchmark data in this repository. All `.py` files byte-compile; the model was
not run here because `mamba-ssm` needs a CUDA GPU.

## What is in the code

| Part | File | What it does |
|---|---|---|
| Mel encoder | `src/models/mel_encoder.py` | `Linear(n_mels → d_model)` + `LayerNorm` |
| Context tower | `src/models/mamba_tower.py` | `mamba_ssm.Mamba` layers; defaults `n_layers=6`, `d_state=32`, `d_conv=4` |
| Parameter head | `src/models/parameter_head.py` | sigmoid-bounded `gain_db` [-40, 0] dB, `attack_ms` [0.1, 100], `release_ms` [10, 2000] |
| Differentiable compressor | `src/models/dsp_layer.py` | IIR attack/release smoothing, lookahead of 4 frames (`lookahead_frames=4`) |
| Full model | `src/models/context_compressor.py` | `ContextAwareCompressor(audio, mel, return_params=False)`; build with `ContextAwareCompressor.from_config(cfg)` |
| Losses | `src/losses/` | multi-resolution STFT, LUFS target, crest factor |
| Data | `src/data/` | blind (synthetic corruption) and paired datasets |
| Training | `scripts/train_blind.py` | PyTorch Lightning blind pre-training |

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
- `src/utils/realtime.py` streaming inference
- Tests, examples, API docs
- Latency or memory benchmarks (none have been measured)
- NVIDIA NeMo integration (not imported anywhere; commented out in `requirements.txt`)

`emotion-aware-dsp` is a git submodule entry with no `.gitmodules`, so it
cannot be checked out.

## Licence

MIT — see [LICENSE](LICENSE). Mamba SSM by Albert Gu and Tri Dao
([state-spaces/mamba](https://github.com/state-spaces/mamba)).

Contact: scott@sgmstudios.ca
