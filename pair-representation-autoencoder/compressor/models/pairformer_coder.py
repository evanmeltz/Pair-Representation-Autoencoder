import torch
import torch.nn as nn
from mmengine.model import BaseModule

from protenix.model.modules.pairformer import PairformerStack


class PairformerPairCoder(BaseModule):
    """Pairformer-based encoder/decoder for pair representation compression.

    A unified class that serves as either encoder or decoder depending on
    the channel_dims configuration. Each stage applies a PairformerStack
    (dimension-preserving) followed by a Linear projection to the next
    dimension.

    For encoding:  channel_dims=[128, 64, 32, 16, 4]
    For decoding:  channel_dims=[4, 16, 32, 64, 128]

    Args:
        channel_dims (list[int]): Sequence of channel dimensions.
            First element is the input dimension, last is the output.
        n_blocks (int | list[int]): Number of PairformerBlock layers per stage.
            Can be a single int (applied to all stages) or a list (per-stage config).
            Default: 2.
        n_heads (int | list[int]): Number of attention heads (unused with c_s=0).
            Can be a single int (applied to all stages) or a list (per-stage config).
            Default: 4.
        num_intermediate_factor (int): FFN expansion factor. Default: 4.
        dropout (float): Dropout rate for triangle operations. Default: 0.0.
        blocks_per_ckpt (int | None): Activation checkpoint granularity. Default: None.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(
        self,
        channel_dims: list[int] = [128, 64, 32, 16, 4],
        n_blocks: int | list[int] = 2,
        n_heads: int | list[int] = 4,
        num_intermediate_factor: int = 4,
        dropout: float = 0.0,
        blocks_per_ckpt: int | None = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert len(channel_dims) >= 2, "channel_dims must have at least 2 elements"

        self.channel_dims = channel_dims
        n_stages = len(channel_dims) - 1

        # Convert single values to lists for all stages
        if isinstance(n_blocks, int):
            n_blocks = [n_blocks] * n_stages
        if isinstance(n_heads, int):
            n_heads = [n_heads] * n_stages

        # Validate list lengths
        assert (
            len(n_blocks) == n_stages
        ), f"n_blocks list length ({len(n_blocks)}) must match number of stages ({n_stages})"
        assert (
            len(n_heads) == n_stages
        ), f"n_heads list length ({len(n_heads)}) must match number of stages ({n_stages})"

        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()

        for i in range(n_stages):
            self.stages.append(
                PairformerStack(
                    n_blocks=n_blocks[i],
                    n_heads=n_heads[i],
                    c_z=channel_dims[i],
                    c_s=0,
                    num_intermediate_factor=num_intermediate_factor,
                    dropout=dropout,
                    blocks_per_ckpt=blocks_per_ckpt,
                    hidden_scale_up=False,
                )
            )
            self.projections.append(nn.Linear(channel_dims[i], channel_dims[i + 1]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, L, C_in) pair representation.

        Returns:
            (B, L, L, C_out) transformed pair representation.
        """
        B, L, _, _ = x.shape

        pairformer_mask = mask.clone()
        # try deleting the diagonal because no padding
        diag_idx = torch.arange(L, device=mask.device)
        pairformer_mask[:, diag_idx, diag_idx] = 1.0

        z = x
        for stack, proj in zip(self.stages, self.projections):
            _, z = stack(
                s=None,
                z=z,
                pair_mask=pairformer_mask,
                triangle_multiplicative="cuequivariance",
                triangle_attention="cuequivariance",
                # triangle_attention="torch",
            )
            z = proj(z)

        return z