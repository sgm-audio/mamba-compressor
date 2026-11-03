"""CPU tests for the differentiable compressor (no mamba-ssm needed)."""

import pytest
import torch

from src.models.dsp_layer import DifferentiableDSPCompressor

SR, HOP, LA_FRAMES = 44100, 512, 4
FRAMES = 40
STEP_FRAME = 20  # gain drops by 20 dB at this frame


def _step_params(fast_ms: float = 0.01):
    g = torch.zeros(1, FRAMES)
    g[0, STEP_FRAME:] = -20.0
    return {
        "gain_db": g,
        "attack_ms": torch.full((1, FRAMES), fast_ms),
        "release_ms": torch.full((1, FRAMES), fast_ms),
    }


def _first_below_half(y):
    return int((y[0] < 0.5).nonzero()[0].item())


def test_lookahead_applies_gain_early():
    """With lookahead L, gain reduction must start about L samples BEFORE the step."""
    comp = DifferentiableDSPCompressor(SR, HOP, LA_FRAMES)
    y = comp(torch.ones(1, FRAMES * HOP), _step_params())
    shift = _first_below_half(y) - STEP_FRAME * HOP
    L = LA_FRAMES * HOP
    # Linear frame->sample interpolation moves the -6 dB crossing by up to one hop.
    assert -L - HOP <= shift <= -L + HOP, f"shift={shift}, expected about {-L}"


def test_no_lookahead_is_not_early():
    comp = DifferentiableDSPCompressor(SR, HOP, 0)
    y = comp(torch.ones(1, FRAMES * HOP), _step_params())
    shift = _first_below_half(y) - STEP_FRAME * HOP
    assert -HOP <= shift <= HOP, f"shift={shift}"


def test_shape_and_finite():
    comp = DifferentiableDSPCompressor(SR, HOP, LA_FRAMES)
    x = torch.randn(2, FRAMES * HOP) * 0.5
    p = _step_params(5.0)
    p = {k: v.expand(2, -1) for k, v in p.items()}
    y = comp(x, p)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_unity_gain_passes_audio_through():
    comp = DifferentiableDSPCompressor(SR, HOP, LA_FRAMES)
    x = torch.randn(1, FRAMES * HOP)
    p = {
        "gain_db": torch.zeros(1, FRAMES),
        "attack_ms": torch.full((1, FRAMES), 5.0),
        "release_ms": torch.full((1, FRAMES), 50.0),
    }
    torch.testing.assert_close(comp(x, p), x)


def test_attack_is_faster_than_release():
    """Gain falls with the attack constant and recovers with the release constant."""
    comp = DifferentiableDSPCompressor(SR, HOP, 0)
    g = torch.zeros(1, FRAMES)
    g[0, 10:20] = -20.0
    p = {
        "gain_db": g,
        "attack_ms": torch.full((1, FRAMES), 1.0),
        "release_ms": torch.full((1, FRAMES), 200.0),
    }
    y = comp(torch.ones(1, FRAMES * HOP), p)[0]
    # 2 ms after the drop the gain is well down; 2 ms after the rise it's still low.
    assert y[10 * HOP + 2 * HOP + 88] < 0.3
    assert y[20 * HOP + HOP + 88] < 0.5


def test_gradients_flow_to_gain():
    comp = DifferentiableDSPCompressor(SR, HOP, LA_FRAMES)
    p = _step_params(5.0)
    p["gain_db"].requires_grad_(True)
    comp(torch.ones(1, FRAMES * HOP), p).sum().backward()
    assert p["gain_db"].grad is not None
    assert torch.isfinite(p["gain_db"].grad).all()
    assert p["gain_db"].grad.abs().sum() > 0


def test_streaming_is_explicitly_unimplemented():
    from src.models.mamba_tower import MambaContextTower

    with pytest.raises(NotImplementedError):
        MambaContextTower.forward_with_state(object(), torch.zeros(1, 4, 8))
