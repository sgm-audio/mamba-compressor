# Training notes

Only blind pre-training is written. It needs a CUDA GPU (`mamba-ssm`,
`causal-conv1d`).

```bash
pip install -r requirements.txt
# set data.audio_dir in configs/blind_pretrain.yaml to a folder of audio files
python scripts/train_blind.py --config configs/blind_pretrain.yaml --gpus 0
```

Flags: `--config`, `--gpus`, `--epochs`, `--batch_size`, `--resume`.

What blind pre-training does (`src/data/blind_dataset.py`): it takes clean audio,
runs it through a conventional compressor with random ratio, threshold, attack
and release, and gives the model the (corrupted, clean) pair. The losses are
multi-resolution STFT, a LUFS target and a crest-factor term (`src/losses/`).

Defaults (`configs/blind_pretrain.yaml`): 44.1 kHz, n_fft 2048, hop 512,
128 mels, 6 Mamba layers, d_model 256, d_state 32, lookahead 4 frames.

No run of this script is recorded in the repo, so there are no loss curves,
checkpoints or timings to report yet.
