# Sanitize manifest: mamba-compressor (2026-10-31)

Branch `chore/sanitize`, cut from `master` @ `7ffab07`. 7 local commits, including this file. Nothing pushed. PR #1 (docs/match-reality @0edfa02) is included as the first commit.

## Removed
| Path | Why |
|---|---|
| `GEMINI.md`, `docs/SELF_ANALYSIS.md`, `docs/SOCIAL_MEDIA.md` | AI-generated notes and social-media drafts. |
| `docs/guides/user_guide.md` | It described `from_pretrained`, a `StreamingInference` class, "sub-5ms latency on NVIDIA GPUs" and a NeMo setup. None of these exist. Replaced by `docs/training.md`. |
| `emotion-aware-dsp` gitlink | No `.gitmodules` entry, and the target repo returns 404 publicly. |

## Added
`pyproject.toml` (ruff/pytest config), `tests/test_dsp_layer.py` (7 CPU tests), `scripts/ci.sh`, `.github/workflows/ci.yml` (calls ci.sh; it won't run until Actions billing is unlocked), `docs/training.md`, and `.gitignore` entries for checkpoints and data. LICENSE was already MIT.

## Code fixes (tested)
- **`dsp_layer.py` lookahead:** padding at the front delayed the gain envelope by `lookahead_samples` instead of advancing it.
  - Step test at 44.1 kHz, hop 512, 4 frames: before, the −6 dB crossing came 2077 samples after the requested change; now it comes 2019 samples before (about 2048 expected).
  - `test_lookahead_applies_gain_early` fails on the old code (`_logs/mcomp-test-on-old-dsp.log`: 1 failed, 6 passed).
- **Imports:** the package imports without mamba-ssm (lazy imports). There's a clear ImportError when mamba-ssm is missing.
- **Streaming:** `forward_with_state()` called `Mamba.forward_with_state`, which mamba_ssm doesn't have (AttributeError on the first chunk). It now raises NotImplementedError, and the README lists streaming as not implemented.

## Doc claims changed
- README usage shows the real `forward(audio, mel)` signature, plus test instructions.
- Streaming is listed as not implemented.
- CONTRIBUTING lost its placeholder links and email and a non-existent `.[dev]` extra.
- The placeholder author and email were removed from `src/__init__.py`.

## Tests, before and after (shared 8-vCPU Intel Xeon VM, Linux, Python with torch on CPU)
| Check | Before (`master` @ 7ffab07) | After |
|---|---|---|
| Tests | none in the repo; importing without mamba-ssm failed | `scripts/ci.sh`: compileall, ruff (errors only) and pytest: 7 passed |
| Lookahead step test | crossing +2077 samples (late) | −2019 samples (early, as intended) |
| `forward` on 20,480 samples, CPU | 313.1 ms | 314.3 ms (no change) |
Logs: `_logs/mcomp-before-repro.log`, `mcomp-after-repro.log`, `mcomp-after-ci.log`, `mcomp-test-on-old-dsp.log`.

## Needs Scott
- **CI billing.**
- **History rewrite:** optional here. There are no local paths in history; it would only drop the three agent/draft files.

## History rewrite (prepared, not run on the real repo)

Nothing has been rewritten or pushed. The commands below were tried on a throwaway `--mirror` clone in `/tmp` on 2026-10-31. They exited 0, and afterwards `git log --all -p` and the commit messages had 0 matches for the local-path patterns. Pack size went from 56 KiB to about the same. That scratch clone was deleted.

The secret scans (gitleaks `git` and `dir`, and trufflehog, over all history) found no secrets. So this rewrite is about local paths and large or unneeded files, not live keys. Scott says any exposed keys were rotated anyway.

`paths-mamba-compressor.txt` (paths removed from every commit):
```text
GEMINI.md
docs/SOCIAL_MEDIA.md
docs/SELF_ANALYSIS.md
```

`replace.txt` is kept outside the repo, so the paths it removes aren't republished here. Use `/workspace/handoff/_history/replace-paths.txt`. It has 8 literal lines covering the Linux, Fedora Atomic (`/var/home`), macOS and Windows home-directory prefixes and the `x:` workspace URI. The `/var/home` line comes before the plain `/home` line.

Commands:
```bash
# 1. Merge chore/sanitize into the default branch first, so the rewrite covers the cleaned tree.
git clone --mirror https://github.com/sgm-audio/mamba-compressor.git mamba-compressor-rewrite.git
cd mamba-compressor-rewrite.git
git filter-repo --invert-paths --paths-from-file ../paths-mamba-compressor.txt --replace-text ../replace.txt
# 2. Check before pushing
git log --all -p | grep -cEf ../path-patterns.txt   # expect 0 (patterns file is next to replace.txt)
gitleaks git . --no-banner --redact                                                       # expect no leaks
git count-objects -vH
# 3. Push. Needs Scott's OK: this rewrites every commit hash.
git push --force --all origin && git push --force --tags origin
```

Caveats:
- GitHub keeps `refs/pull/*` for every PR, and a force-push can't touch them. The old objects stay reachable through those PR refs and cached commit URLs until GitHub Support runs a GC; ask them to purge after the push.
- Forks and existing clones keep the old history. Collaborators need to re-clone, and open PRs have to be recreated.
