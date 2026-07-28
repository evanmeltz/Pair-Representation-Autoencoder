import torch
import torch.nn.functional as F


def kd_loss(pred_logits: torch.Tensor, target_logits: torch.Tensor, temperature: float = 1.0):
    target_probs = F.softmax(target_logits / temperature, dim=-1)
    pred_log_probs = F.log_softmax(pred_logits / temperature, dim=-1)

    # Returns per-pair KL, shape (...), not reduced.
    return F.kl_div(
        pred_log_probs,
        target_probs,
        reduction="none",
    ).sum(dim=-1) * (temperature ** 2)


def logits_to_distances(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return (probs * bin_centers).sum(dim=-1)