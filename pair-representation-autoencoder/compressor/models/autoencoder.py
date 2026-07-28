from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel

from mmengine.registry import MODELS

from compressor.models.distogram_head import FF3DistogramHead

class RowColumnTransformerBlock(nn.Module):
    """Apply row and column self-attention in parallel, then a 4x MLP."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def _attend_by_batches(
        self,
        x: torch.Tensor,
        batches: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run MHA over precomputed same-length row/column batches.
        Args:
            x: (N, C)
            batches: list of LongTensor, each shaped (B_groups, seq_len).
                Each row in the tensor is one row/column's flattened indices.
        Returns:
            out: (N, C)
        """
        out = torch.zeros_like(x)

        for idx in batches:
            # idx: (num_groups, seq_len)
            y_in = x[idx]  # (num_groups, seq_len, C)
            y, _ = self.attn(y_in, y_in, y_in, need_weights=False,)
            out[idx] = y.to(dtype=out.dtype)

        return out

    def forward(
        self,
        x: torch.Tensor,
        attention_batches: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        
        row_out = self._attend_by_batches(x, attention_batches["rows"])
        col_out = self._attend_by_batches(x, attention_batches["cols"])
        x = self.norm1(x + row_out + col_out)
        return self.norm2(x + self.ffn(x))


class RowColumnTransformerStage(nn.Module):
    """Concat spatial encoding, run row/column transformer blocks, project channels."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spatial_encoding_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim = in_dim + spatial_encoding_dim
        self.blocks = nn.ModuleList(
            [RowColumnTransformerBlock(dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.project = nn.Linear(dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        spatial_encoding: torch.Tensor,
        attention_batches: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        x = torch.cat([x, spatial_encoding], dim=-1)

        for block in self.blocks:
            x = block(x, attention_batches)

        return self.project(x)


@MODELS.register_module()
class PairAutoencoder(BaseModel):
    """Row/column-attention autoencoder for pair representation compression.

    Compresses ``(L, L, in_channels)`` pair features to
    ``(L, L, bottleneck_dim)`` and reconstructs the original channel dimension.
    Row and column self-attention are applied in parallel at each transformer
    block. Training uses reconstruction losses and a frozen distogram teacher.

    This implementation supports batch size one.

    Args:
        in_channels: Input pair-feature channel dimension.
        bottleneck_dim: Compressed channel dimension.
        encoder_channel_dims: Encoder stage dimensions.
        decoder_channel_dims: Decoder stage dimensions.
        dropout: Dropout used in attention and feed-forward layers.
        num_transformer_blocks: Number of transformer blocks per stage.
        num_attention_heads: Number of attention heads per block.
        spatial_encoding_dim: Spatial encoding dimension. Must be 4.
        distogram_head_weight_path: Path to pretrained distogram-head weights.
        normalization_stats_path: Path to a checkpoint containing ``mean`` and
            ``std`` tensors with shape ``(in_channels,)``.
        loss_weights: Weights for ``mse``, ``cosine``, and ``distogram`` losses.
        num_dist_bins: Number of distogram output bins.
    """

    def __init__(
        self,
        in_channels: int = 128,
        bottleneck_dim: int = 8,
        
        encoder_channel_dims: list | None = None,
        decoder_channel_dims: list | None = None,
        dropout: float = 0.0,
        num_transformer_blocks: int = 2,
        num_attention_heads: int = 4,
        spatial_encoding_dim: int = 4,
        
        distogram_head_weight_path: str = "",
        normalization_stats_path: str = "",
        loss_weights: dict = None,
        num_dist_bins: int = 64,
        init_cfg=None,
        data_preprocessor=None,
    ):
        super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if loss_weights is None:
            loss_weights = dict(mse=1.0, cosine=1.0, distogram=1.0)
        self.loss_weights = loss_weights
        self.num_dist_bins = num_dist_bins
        
        if encoder_channel_dims is None:
            encoder_channel_dims = [in_channels, 64, 32, 16, bottleneck_dim]
        if decoder_channel_dims is None:
            decoder_channel_dims = [bottleneck_dim, 16, 32, 64, in_channels]
        
        self.spatial_encoding_dim = spatial_encoding_dim
        self.encoder = self._build_transformer_stages(
            encoder_channel_dims,
            spatial_encoding_dim=spatial_encoding_dim,
            num_blocks=num_transformer_blocks,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.decoder = self._build_transformer_stages(
            decoder_channel_dims,
            spatial_encoding_dim=spatial_encoding_dim,
            num_blocks=num_transformer_blocks,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Frozen distogram head: Linear(in_channels, num_dist_classes)
        self.distogram_head = FF3DistogramHead(
            pair_channel=in_channels,
            num_bins=num_dist_bins,
            weight_path=distogram_head_weight_path,
        )

        # normalize
        self.register_buffer("mean", torch.tensor([0.0] * in_channels))
        self.register_buffer("std", torch.tensor([1.0] * in_channels))
        self.load_mean_std(normalization_stats_path, in_channels)

    def load_mean_std(self, stats_path: str, in_channels: int) -> None:
        if not stats_path:
            raise ValueError(
                "normalization_stats_path is required and must point to a checkpoint "
                "containing 'mean' and 'std' tensors."
            )

        if not Path(stats_path).is_file():
            raise ValueError(
                f"Normalization statistics file does not exist: {stats_path}"
            )

        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        if "mean" not in stats or "std" not in stats:
            raise ValueError(
                "Normalization statistics checkpoint must contain 'mean' and 'std' tensors."
            )

        mean = stats["mean"].float()
        std = stats["std"].float()
        expected_shape = (in_channels,)
        if tuple(mean.shape) != expected_shape or tuple(std.shape) != expected_shape:
            raise ValueError(
                "Normalization statistics must each have shape "
                f"{expected_shape}; got mean={tuple(mean.shape)}, std={tuple(std.shape)}."
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Normalization statistics must contain only finite values.")
        if not torch.all(std > 0):
            raise ValueError("Normalization standard deviations must all be greater than zero.")

        self.mean.copy_(mean)
        self.std.copy_(std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input tensor."""
        return x * self.std + self.mean
    
    def _flatten_valid_cells(
        self,
        z_ii: torch.Tensor,
        pair_mask: torch.Tensor,
    ):
        """Flatten valid spatial cells into an (N, C) tensor.

        Args:
            z_ii: (B, L, L, C), normalized input.
            pair_mask: (B, L, L), 1 for valid cells, 0 for invalid cells.

        Returns:
            z_flat: (N, C)
            coords: (N, 3), each row is (b, i, j)
        """
        B, L1, L2, C = z_ii.shape

        # Check that batch size is always 1.
        if B != 1:
            raise ValueError(f"This autoencoder expects batch size 1, got B={B}")

        valid_mask = pair_mask.bool()  # (B, L, L)
        coords = valid_mask.nonzero(as_tuple=False)  # (N, 3): b, i, j

        if coords.numel() == 0:
            z_flat = z_ii.new_zeros((0, C))
            return z_flat, coords

        z_flat = z_ii[coords[:, 0], coords[:, 1], coords[:, 2], :]  # (N, C)

        return z_flat, coords
    
    def _scatter_valid_cells(
        self,
        z_flat_reconstructed: torch.Tensor,
        coords: torch.Tensor,
        output_shape: torch.Size,
    ) -> torch.Tensor:
        """Scatter reconstructed valid rows back into original (B, L, L, C) shape."""
        z_reconstructed = z_flat_reconstructed.new_zeros(output_shape)

        if coords.numel() == 0:
            return z_reconstructed

        z_reconstructed[
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            :,
        ] = z_flat_reconstructed

        return z_reconstructed
    
    def _spatial_encoding(
        self,
        coords: torch.Tensor,
        L1: int,
        L2: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a 4D spatial encoding from stored flattened coordinates.

        coords rows are (batch_index, i, j). For B=1, only i and j matter.
        Returns (N, 4): normalized i, normalized j, signed offset, absolute offset.
        """
        if coords.numel() == 0:
            return coords.new_zeros((0, self.spatial_encoding_dim), dtype=dtype)

        i = coords[:, 1].to(dtype=dtype)
        j = coords[:, 2].to(dtype=dtype)
        denom_i = max(L1 - 1, 1)
        denom_j = max(L2 - 1, 1)
        denom = max(max(L1, L2) - 1, 1)

        enc = torch.stack(
            [
                i / denom_i,
                j / denom_j,
                (i - j) / denom,
                (i - j).abs() / denom,
            ],
            dim=-1,
        )

        if self.spatial_encoding_dim != 4:
            raise ValueError("This implementation expects spatial_encoding_dim=4.")
        return enc
    
    def _build_attention_batches(
        self,
        coords: torch.Tensor,
    ) -> dict[str, list[torch.Tensor]]:
        """Precompute same-length row and column batches for batched MHA.
        coords: (N, 3), each row is (batch_index, i, j).
        Returns:
            {
                "rows": list of LongTensor shaped (num_rows_in_batch, row_len),
                "cols": list of LongTensor shaped (num_cols_in_batch, col_len),
            }

        """
        if coords.numel() == 0:
            return {"rows": [], "cols": []}

        def make_batches(group_ids: torch.Tensor) -> list[torch.Tensor]:
            batches_by_len: dict[int, list[torch.Tensor]] = {}

            for group_id in group_ids.unique(sorted=True):
                idx = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
                seq_len = int(idx.numel())

                if seq_len == 0:
                    continue

                if seq_len not in batches_by_len:
                    batches_by_len[seq_len] = []

                batches_by_len[seq_len].append(idx)

            batches: list[torch.Tensor] = []

            for seq_len in sorted(batches_by_len):
                # Each element has the same length, so stack into:
                # (num_groups_with_this_len, seq_len)
                batches.append(torch.stack(batches_by_len[seq_len], dim=0))

            return batches

        row_batches = make_batches(coords[:, 1])
        col_batches = make_batches(coords[:, 2])

        return {
            "rows": row_batches,
            "cols": col_batches,
        }

    def _build_transformer_stages(
        self,
        dims: list[int],
        spatial_encoding_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [
                RowColumnTransformerStage(
                    in_dim=dims[i],
                    out_dim=dims[i + 1],
                    spatial_encoding_dim=spatial_encoding_dim,
                    num_blocks=num_blocks,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def encode(
        self,
        z_flat: torch.Tensor,
        spatial_encoding: torch.Tensor,
        attention_batches: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        for stage in self.encoder:
            z_flat = stage(z_flat, spatial_encoding, attention_batches)
        return z_flat

    def decode(
        self,
        z_compressed: torch.Tensor,
        spatial_encoding: torch.Tensor,
        attention_batches: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        for stage in self.decoder:
            z_compressed = stage(z_compressed, spatial_encoding, attention_batches)
        return z_compressed
    
    def _build_pair_mask(self, z_ii_raw: torch.Tensor) -> torch.Tensor:
        # z_ii: (B, L, L, C)
        # mask is 0 where all channels = 0
        valid = (z_ii_raw != 0).any(dim=-1)   # (B, L, L), bool
        return valid.to(dtype=z_ii_raw.dtype)
    
    def forward(self, *, Z_II, mode="loss", data_sample=None, **kwargs):
        """Forward dispatch following MMEngine BaseModel pattern.

        Args:
            Z_II: (B, L, L, 128) or (L, L, 128) pair representation
            distogram: (B, L, L) or (L, L) ground truth distogram
            mode: 'loss' for training, 'predict' for validation
    """
        z_ii_raw = Z_II
        
        # Ensure batch dimension
        if z_ii_raw.dim() == 3:
            z_ii_raw = z_ii_raw.unsqueeze(0)
        
        # masking where all channels = 0
        pair_mask = self._build_pair_mask(z_ii_raw)

        # normalize Z_II / z_ii_raw
        z_ii = self.normalize(z_ii_raw)
        
        # Explicitly zero invalid positions in normalized space
        pair_mask_expanded = pair_mask.unsqueeze(-1)  # (B, L, L, 1)
        z_ii = z_ii * pair_mask_expanded

        # Flatten valid cells: (B, L, L, C) -> (N, C)
        z_flat, coords = self._flatten_valid_cells(z_ii, pair_mask)

        # Spatial encoding for the valid flattened cells: (N, 4)
        spatial_encoding = self._spatial_encoding(
            coords=coords,
            L1=z_ii.shape[1],
            L2=z_ii.shape[2],
            dtype=z_flat.dtype,
        )

        attention_batches = self._build_attention_batches(coords)

        # Encode/decode z_flat with transformer stages
        z_compressed = self.encode(
            z_flat,
            spatial_encoding,
            attention_batches,
        )      # (N, 4)
        z_flat_reconstructed = self.decode(
            z_compressed,
            spatial_encoding,
            attention_batches,
        )  # (N, 128)
        
        # Scatter back to original shape: (N, C) -> (B, L, L, C)
        z_reconstructed = self._scatter_valid_cells(
            z_flat_reconstructed=z_flat_reconstructed,
            coords=coords,
            output_shape=z_ii.shape,
        )

        # Keep invalid regions zero in normalized space
        z_reconstructed = z_reconstructed * pair_mask_expanded

        z_reconstructed_denorm = self.denormalize(z_reconstructed)
        # mask intrachain to 0
        z_reconstructed_denorm = z_reconstructed_denorm * pair_mask_expanded
        # pred_logits = self.distogram_head(z_reconstructed_denorm)  # (B, L, L, C_bins)
        
        distogram_out = self.distogram_head(
            x_gt=z_ii_raw,
            x_pred=z_reconstructed_denorm,
            mask=pair_mask,
        )

        
        if mode == "loss":
            # get dimensions of normalized input (B, L_max, L_max, C)
            B, L1, L2, C = z_ii.shape
            
            valid_mask = pair_mask.bool()  # (B, L, L)
            valid_mask_expanded = valid_mask.unsqueeze(-1)  # (B, L, L, 1)

            # Masked MSE loss
            # mse_loss = self.mse_loss_f(z_reconstructed, z_ii)
            sq_err = (z_reconstructed - z_ii) ** 2
            sq_err = sq_err * valid_mask_expanded
            denom = valid_mask_expanded.sum() * C
            mse_loss = sq_err.sum() / denom.clamp_min(1.0)

            # Masked cosine similarity loss: 1 - cos_sim
            z_loss_flat = z_ii.reshape(B, -1, C)
            z_recon_loss_flat = z_reconstructed.reshape(B, -1, C)
            valid_flat = valid_mask.reshape(B, -1)
            
            cos_sim = F.cosine_similarity(z_recon_loss_flat, z_loss_flat, dim=-1) # (B, L*L)
            cosine_loss = (1.0 - cos_sim)[valid_flat].mean()

            distogram_loss = distogram_out["loss"]

            total_loss = (
                self.loss_weights["mse"] * mse_loss
                + self.loss_weights["cosine"] * cosine_loss
                + self.loss_weights["distogram"] * distogram_loss
            )

            return dict(
                loss=total_loss,
                mse_loss=mse_loss.detach(),
                cosine_loss=cosine_loss.detach(),
                distogram_loss=distogram_loss.detach(),
            )

        elif mode == "predict":
            # Convert logits to expected distances for evaluation
            # pred_distances = self.logits_to_distances(pred_logits)
            output_list = []
            # for i in range(Z_II.shape[0]):
            for i in range(z_ii.shape[0]):
                output_list.append(
                    dict(
                        Z_II_original=z_ii_raw[i],
                        Z_II_reconstructed=z_reconstructed_denorm[i],
                        pair_mask=pair_mask[i],
                        distogram_gt=distogram_out["gt_distances"][i],
                        distogram_pred=distogram_out["pred_distances"][i],
                    )
                )
            return output_list

        else:
            raise ValueError(f"Unknown mode: {mode}")