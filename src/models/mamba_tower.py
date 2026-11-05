"""Mamba Context Tower for temporal semantic reasoning."""

import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:  # mamba-ssm needs a CUDA build; DSP-only use and CPU tests don't need it
    Mamba = None
from typing import Optional


class MambaContextTower(nn.Module):
    """
    Stack of Mamba SSM layers for temporal context modeling.
    
    Each layer has O(T) complexity and carries a hidden state that encodes
    everything heard so far. This IS the context.
    
    d_state=32 gives ~4s of effective temporal memory at hop=512, sr=44100.
    Increase to 64 for complex classical/jazz arrangement awareness.
    
    Args:
        d_model: Model dimension (default: 256)
        n_layers: Number of Mamba layers (default: 6)
        d_state: SSM state size (default: 32)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "MambaContextTower needs mamba-ssm (CUDA build). Install it with "
                "`pip install mamba-ssm causal-conv1d` on a machine with a CUDA GPU."
            )
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                ),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba tower.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
        
        Returns:
            Context-enriched tensor of shape (B, T, d_model)
        """
        # x: (B, T, d_model)
        for layer in self.layers:
            mamba, norm, dropout = layer[0], layer[1], layer[2]
            x = x + mamba(x)   # Residual connection — critical for gradient flow
            x = norm(x)
            x = dropout(x)
        return x
    
    def forward_with_state(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Forward pass with explicit state management for streaming inference.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            states: List of previous states (one per layer), or None for first chunk
        
        Returns:
            Tuple of (output tensor, new states)
        """
        # mamba_ssm.Mamba has no forward_with_state(); the previous code called it
        # and would raise AttributeError on the first chunk. Streaming needs
        # Mamba.step() + allocate_inference_cache() and a GPU to verify, so it
        # is not implemented yet.
        raise NotImplementedError(
            "Streaming inference is not implemented. Use forward() on whole clips."
        )
