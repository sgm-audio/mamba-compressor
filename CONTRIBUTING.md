# Contributing

Issues and pull requests are welcome.

- Bug reports: Python and PyTorch versions, GPU (if any), the traceback, and the
  smallest input that reproduces it.
- Code: run `scripts/ci.sh` before opening a PR and paste its last lines in the
  PR description. Bug fixes should come with a test that fails without the fix.
- Tests that need `mamba-ssm` (CUDA) can't run in CPU CI yet; say in the PR
  where and how you ran them.

Open work is listed under "Not implemented yet" in the README.

Contributions are licensed under the MIT License (see `LICENSE`).
