from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
from mmengine.registry import METRICS


@METRICS.register_module()
class CompressorEvaluator(BaseMetric):
    """Evaluator for the pair representation autoencoder.

    Computes:
    - L1 error on reconstructed vs original Z_II
    - Cosine similarity on reconstructed vs original Z_II
    - L1 error on predicted vs ground truth distogram
    """

    default_prefix = "compressor"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: Dict[str, list], data_samples: Sequence[dict]):
        """Process one batch of data samples.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model (mode='predict').
        """
        for output_data in data_samples:
            z_orig = output_data["Z_II_original"]  # (L, L, 128)
            z_recon = output_data["Z_II_reconstructed"]  # (L, L, 128)
            dg_gt = output_data["distogram_gt"]      # (L, L)
            dg_pred = output_data["distogram_pred"]  # (L, L)
            pair_mask = output_data["pair_mask"].bool()    # (L, L)
            
            valid_mask_expanded = pair_mask.unsqueeze(-1)  # (L, L, 1)

            # masked l1
            abs_err = (z_recon - z_orig).abs() * valid_mask_expanded
            denom = valid_mask_expanded.sum().item() * z_orig.shape[-1]
            z_ii_l1 = (abs_err.sum().item() / max(denom, 1.0))

            # Masked cosine similarity on Z_II (per-position, then average)
            C = z_orig.shape[-1]
            z_flat_orig = z_orig.reshape(-1, C)
            z_flat_recon = z_recon.reshape(-1, C)
            valid_flat = pair_mask.reshape(-1)

            cos_vals = F.cosine_similarity(z_flat_recon, z_flat_orig, dim=-1)
            cos_vals = cos_vals[valid_flat]
            cos_sim = cos_vals.mean().item() if cos_vals.numel() > 0 else 0.0

            # L1 on distogram
            valid = pair_mask.bool()
            distogram_l1 = (
                (dg_pred - dg_gt).abs()[valid].mean().item()
            )

            # Only store scalar metrics, not the full tensors
            result = dict(
                z_ii_l1=z_ii_l1,
                z_ii_cosine_sim=cos_sim,
                distogram_l1=distogram_l1,
            )
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute aggregated metrics from processed results.

        Returns:
            dict with keys: z_ii_l1, z_ii_cosine_sim, distogram_l1
        """
        # Results now contain pre-computed scalar metrics, just average them
        z_ii_l1_list = [res["z_ii_l1"] for res in results]
        cosine_sim_list = [res["z_ii_cosine_sim"] for res in results]
        distogram_l1_list = [res["distogram_l1"] for res in results]

        metrics = {
            "z_ii_l1": sum(z_ii_l1_list) / len(z_ii_l1_list),
            "z_ii_cosine_sim": sum(cosine_sim_list) / len(cosine_sim_list),
            "distogram_l1": sum(distogram_l1_list) / len(distogram_l1_list),
        }
        return metrics
