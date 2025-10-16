"""FGSM and PGD Implementation
"""
import torch
import torch.nn as nn


# ImageNet normalization constants (must match data.py)
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denorm(x_norm: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensor to pixel space [0,1]."""
    mean = imagenet_mean.to(x_norm.device)
    std = imagenet_std.to(x_norm.device)
    denormed = x_norm * std + mean
    return denormed


def renorm(x_pix: torch.Tensor) -> torch.Tensor:
    """Convert pixel space [0,1] to normalized tensor."""
    mean = imagenet_mean.to(x_pix.device)
    std = imagenet_std.to(x_pix.device)
    renormalized = (x_pix - mean) / std
    return renormalized


def fgsm(model: nn.Module, x_norm: torch.Tensor, y: torch.Tensor, eps_pix: float) -> torch.Tensor:
    """Fast Gradient Sign Method (FGSM) in pixel space."""
    model.eval()
    
    # convert to pixel space
    x_pix = denorm(x_norm).detach()
    x_pix.requires_grad_(True)
    
    # forward pass on renormalized input
    logits = model(renorm(x_pix))
    loss = nn.CrossEntropyLoss()(logits, y)
    
    loss.backward()
    grad_sign = x_pix.grad.sign()
    
    # application of FGSM perturbation and clip to [0,1]
    x_adv = torch.clamp(x_pix + eps_pix * grad_sign, 0, 1)
    
    # normalized adversarial example
    x_adv = renorm(x_adv)
    return x_adv.detach()  # we don't need gradient tracking


def pgd(model: nn.Module, x_norm: torch.Tensor, y: torch.Tensor, eps_pix: float, steps: int = 20) -> torch.Tensor:
    """Projected Gradient Descent (PGD) in pixel space."""
    model.eval()
    
    alpha_pix = eps_pix / 4.0
    x0 = denorm(x_norm).detach()
    
    # random start within eps ball
    x = x0 + torch.empty_like(x0).uniform_(-eps_pix, eps_pix)
    x = torch.clamp(x, 0, 1)
    
    # PGD iteration
    for _ in range(steps):
        x.requires_grad_(True)
        
        logits = model(renorm(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        
        loss.backward()
        grad_sign = x.grad.sign()
        
        #v  projection onto L-inf ball around x0
        x = x.detach() + alpha_pix * grad_sign
        x = torch.max(torch.min(x, x0 + eps_pix), x0 - eps_pix)
        x = torch.clamp(x, 0, 1)
    return renorm(x).detach()


