import torch
import torch.nn.functional as F


def multitask_loss(logits, log_magnitude, labels, magnitude, config):
    """Focal classification + positive-only MSE in log1p magnitude (Sect. 2.5.2)."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt = torch.exp(-bce)
    alpha = labels * config.focal_alpha + (1-labels) * (1-config.focal_alpha)
    classification = (alpha * (1-pt).pow(config.focal_gamma) * bce).mean()
    positive = labels == 1
    regression = (F.mse_loss(log_magnitude[positive], torch.log1p(magnitude[positive]))
                  if positive.any() else log_magnitude.sum() * 0)
    return classification + config.regression_weight * regression, classification, regression
