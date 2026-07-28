from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from .distogram_utils import kd_loss, logits_to_distances


class FF3DistogramHead(BaseModule):

    def __init__(
        self,
        pair_channel: int = 128,
        num_bins: int = 64,
        first_break: float = 2.3125,
        last_break: float = 21.6875,
        weight_path: str = "",
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_bins = num_bins
        self.first_break = first_break
        self.last_break = last_break

        if not weight_path:
            raise ValueError(
                "weight_path is required and must point to pretrained distogram-head weights."
            )
        if not Path(weight_path).is_file():
            raise ValueError(
                f"Distogram-head weight file does not exist: {weight_path}"
            )

        self.weight = nn.Parameter(torch.zeros(pair_channel, num_bins))

        ckpt = torch.load(weight_path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            self.weight.copy_(ckpt["half_logits_weight"])

        for param in self.parameters():
            param.requires_grad = False

    def _get_bin_centers(self, device, dtype) -> torch.Tensor:
        breaks = torch.linspace(
            self.first_break, self.last_break, self.num_bins - 1,
            dtype=dtype, device=device,
        )
        bin_width = breaks[-1] - breaks[-2]
        return torch.cat([
            (breaks[0] - bin_width / 2).unsqueeze(0),
            (breaks[:-1] + breaks[1:]) / 2,
            (breaks[-1] + bin_width / 2).unsqueeze(0),
        ])

    def _logits_to_distances(self, logits: torch.Tensor) -> torch.Tensor:
        bin_centers = self._get_bin_centers(logits.device, logits.dtype)
        return logits_to_distances(logits, bin_centers)

    def _symmetrized_logits(self, x: torch.Tensor) -> torch.Tensor:
        half = torch.einsum('...a,ab->...b', x, self.weight)
        return half + half.transpose(-2, -3)

    def forward(
        self,
        x_gt: torch.Tensor,
        x_pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """Compute distogram predictions and KD loss.

        Args:
            x_gt:   (..., pair_channel) — original (GT) pair features
            x_pred: (..., pair_channel) — reconstructed pair features
            mask:   (...) float tensor of valid positions

        Returns:
            dict with keys:
                'gt_distances':   (...) expected distances from GT logits
                'pred_distances': (...) expected distances from pred logits
                'loss':           scalar masked KD loss
        """
        with torch.no_grad():
            gt_logits = self._symmetrized_logits(x_gt)
            gt_distances = self._logits_to_distances(gt_logits)

        pred_logits = self._symmetrized_logits(x_pred)
        pred_distances = self._logits_to_distances(pred_logits)

        result = dict(gt_distances=gt_distances, pred_distances=pred_distances)

        loss = kd_loss(
            pred_logits, gt_logits, temperature=1.0
        )
        loss = loss[mask.bool()].mean()
        result["loss"] = loss

        return result