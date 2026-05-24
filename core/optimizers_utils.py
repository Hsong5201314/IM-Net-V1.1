"""
Code Availability: Model-agnostic Proactive meta-control resolves dynamical instability in multi-objective learning via spectral regularization
File: core/optimizers_utils.py
Description:
    High-performance scientific utilities for spectral analysis of the loss
    landscape and gradient dynamics. Implements matrix-free Power Iteration
    for Hessian spectral radius estimation and gradient conflict tracking.
Date: 2026-02-14
"""

import torch
import numpy as np
import torch.nn.functional as F


def compute_hessian_spectral_radius(model, loss, num_iter=10, tolerance=1e-6):
    """
    Quantifies the maximum eigenvalue (lambda_max) of the Hessian matrix.

    Mathematical Basis:
    Matrix-free Power Iteration via Hessian-Vector Product (HVP).
    This computes the 'sharpness' of the optimization manifold, enabling
    the Proactive Meta-Control (IM-Net) to regularize spectral instability.

    Args:
        model: The neural manifold encoder (e.g., LightGCN).
        loss: The scalar energy functional (Weighted Loss).
        num_iter: Iterations for spectral convergence.
        tolerance: Numerical stability term to prevent division by zero.

    Returns:
        float: The spectral radius (maximum curvature) of the Hessian.
    """
    # 1. Isolate the differentiable manifold
    params = [p for p in model.parameters() if p.requires_grad]

    # 2. Compute first-order gradient field
    # create_graph=True is essential for higher-order derivatives
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    def flatten_tensors(tensors):
        return torch.cat([t.contiguous().view(-1) for t in tensors])

    flat_grads = flatten_tensors(grads)

    # 3. Initialize curvature probing vector (v)
    v = torch.randn(flat_grads.size()).to(flat_grads.device)
    v = v / (torch.norm(v) + tolerance)

    # 4. Iterative Spectral Refinement
    for _ in range(num_iter):
        # Implicit Hessian-Vector Product: H * v = ∇(∇L · v)
        gv_product = torch.sum(flat_grads * v)

        # Second-order differentiation
        hvp_tensors = torch.autograd.grad(gv_product, params, retain_graph=True)
        hvp = flatten_tensors(hvp_tensors)

        # Update and normalize the principal eigenvector
        v = hvp / (torch.norm(hvp) + tolerance)

        # Crucial for memory: detach the eigenvector from the graph
        v = v.detach()

    # 5. Rayleigh Quotient: λ_max ≈ (v.T * H * v)
    gv_product = torch.sum(flat_grads * v)
    hvp_tensors = torch.autograd.grad(gv_product, params, retain_graph=True)
    hvp = flatten_tensors(hvp_tensors)

    spectral_radius = torch.dot(v, hvp).item()

    # 6. Explicit Memory Reclamation
    del flat_grads
    del hvp

    return abs(spectral_radius)


def compute_gradient_conflict_metrics(loss_main, loss_aux, model, retain_graph=True):
    """
    Computes the gradient conflict (Cosine Similarity) between the main task
    and the auxiliary (harmful) task.

    Mathematical meaning:
    rho = cos(theta) = <g_main, g_aux> / (||g_main|| * ||g_aux||)
    If rho < 0, the gradients are conflicting (interfering).
    If rho < -0.9, it's severe adversarial interference (simulation mode target).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # Obtain the main task gradient
    grad_main = torch.autograd.grad(
        loss_main, params, retain_graph=retain_graph, allow_unused=True
    )
    # Obtain gradients of auxiliary and conflicting tasks
    grad_aux = torch.autograd.grad(
        loss_aux, params, retain_graph=retain_graph, allow_unused=True
    )

    # Flatten into a 1D vector (filter out parameters without gradients)
    flat_g_main = torch.cat([g.contiguous().view(-1) for g in grad_main if g is not None])
    flat_g_aux = torch.cat([g.contiguous().view(-1) for g in grad_aux if g is not None])

    # Safety check: prevent empty gradients
    if flat_g_main.numel() == 0 or flat_g_aux.numel() == 0:
        return 0.0

    # Calculate cosine similarity ρ
    cos_sim = F.cosine_similarity(flat_g_main, flat_g_aux, dim=0).item()

    # Explicitly free the GPU memory occupied by the computation graph
    del grad_main, grad_aux, flat_g_main, flat_g_aux

    return cos_sim


class HessianTracker:
    """
    A stateful tracker for managing intermittent Hessian spectral radius calculations.

    Why it's needed:
    Computing the Hessian spectral radius via Power Iteration requires double
    backpropagation, which is computationally expensive. This tracker acts as a
    controller to only compute it every `compute_freq` steps, preventing training
    slowdowns while still capturing the topological phase transitions.
    """

    def __init__(self, model, compute_freq=10, num_iter=20, loss_scaling=1.0):
        self.model = model
        self.compute_freq = compute_freq
        self.num_iter = num_iter
        self.loss_scaling = loss_scaling
        self.step_count = 0
        self.last_computed_radius = 0.0

    def step_and_compute(self, loss):
        """
        Call this in your training loop. It will evaluate the spectral radius
        only at the specified frequency intervals.
        """
        self.step_count += 1

        # Trigger second-order differentiation when reaching the calculation frequency
        if self.step_count % self.compute_freq == 0:
            scaled_loss = loss * self.loss_scaling

            # Call the operators defined at the top of this file
            radius = compute_hessian_spectral_radius(
                model=self.model,
                loss=scaled_loss,
                num_iter=self.num_iter
            )
            self.last_computed_radius = radius
            return radius

        # Otherwise directly return the previously calculated value without consuming computing resources
        return self.last_computed_radius
