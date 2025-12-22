"""Offline probing utilities for computing CAFE and FedZero statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from learners.learner import Learner


@dataclass
class ClientStats:
    """Per-client gradients and loss statistics used by CAFE and FedZero."""

    client_id: int
    gradient: torch.Tensor
    num_probe_samples: int
    sum_squared_losses: float
    sigma: float


def _build_probe_loader(
    full_loader: DataLoader,
    probe_fraction: float = 0.05,
    min_samples: int = 32,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> DataLoader:
    """Return a loader sampling a deterministic probe subset of ``full_loader``.

    The probe loader reuses ``full_loader``'s dataset and batch size but relies on
    a :class:`~torch.utils.data.SubsetRandomSampler` built from a fixed seed. The
    sample count is derived from ``probe_fraction`` clamped between
    ``min_samples`` and ``max_samples`` (if provided).
    """

    if not (0.0 < probe_fraction <= 1.0):
        raise ValueError(f"probe_fraction must be in (0, 1], got {probe_fraction}")

    dataset = full_loader.dataset
    total_examples = len(dataset)
    if total_examples == 0:
        raise ValueError("Cannot build a probe loader from an empty dataset.")

    probe_size = max(min_samples, int(total_examples * probe_fraction))
    if max_samples is not None:
        probe_size = min(probe_size, max_samples)
    probe_size = min(probe_size, total_examples)

    generator = torch.Generator().manual_seed(seed)
    probe_indices = torch.randperm(total_examples, generator=generator)[:probe_size].tolist()

    sampler = SubsetRandomSampler(probe_indices)
    return DataLoader(
        dataset,
        batch_size=full_loader.batch_size,
        sampler=sampler,
        drop_last=False,
    )


def _compute_client_probe_stats(
    learner: Learner,
    probe_loader: DataLoader,
) -> Tuple[torch.Tensor, int, float]:
    """Collect gradient and loss statistics on a probe set for one client.

    Returns a triple of (average gradient, number of samples, sum of squared
    losses). The learner is evaluated in inference mode and expects
    ``learner.criterion`` to output unreduced per-sample losses.
    """

    device = learner.device
    model = learner.model
    criterion = learner.criterion

    if criterion is None:
        raise ValueError("Learner.criterion must not be None to compute gradients.")

    model.eval()

    num_samples = 0
    sum_squared_losses = 0.0
    gradient_accumulator = torch.zeros(learner.model_dim, device=device)

    for batch_inputs, batch_targets, _ in probe_loader:
        inputs = batch_inputs.to(device=device, dtype=torch.float32)
        targets = batch_targets.to(device)

        if learner.is_binary_classification:
            targets = targets.to(dtype=torch.float32).unsqueeze(1)

        batch_size = targets.size(0)
        num_samples += batch_size

        model.zero_grad(set_to_none=True)
        predictions = model(inputs)

        losses = criterion(predictions, targets)
        losses = losses.view(batch_size, -1).mean(dim=1)

        sum_squared_losses += losses.detach().pow(2).sum().item()

        losses.sum().backward()
        gradient_accumulator += learner.get_grad_tensor()

    if num_samples == 0:
        raise RuntimeError("Probe loader is empty; cannot compute gradients.")

    mean_gradient = gradient_accumulator / float(num_samples)
    return mean_gradient.detach().clone(), num_samples, float(sum_squared_losses)


def _pairwise_l2_distances(vectors: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L2 distances between rows of ``vectors``."""

    with torch.no_grad():
        squared_norm = (vectors**2).sum(dim=1, keepdim=True)
        squared_distance = squared_norm + squared_norm.t() - 2.0 * (vectors @ vectors.t())
        squared_distance.clamp_(min=0.0)
        return torch.sqrt(squared_distance)


def compute_cafe_and_fedzero_stats(
    reference_learner: Learner,
    clients: Sequence,
    probe_fraction: float = 0.05,
    min_probe_samples: int = 32,
    max_probe_samples: Optional[int] = None,
    base_seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[ClientStats]]:
    """Compute CAFE distance matrix and FedZero utilities for a client set."""

    if not clients:
        raise ValueError("`clients` list is empty; nothing to compute.")

    device = reference_learner.device
    reference_learner.model.eval()

    gradients: List[torch.Tensor] = []
    stats_list: List[ClientStats] = []

    for client_index, client in enumerate(clients):
        train_loader = getattr(client, "train_iterator", None)
        if train_loader is None:
            continue

        client_id = getattr(client, "id", client_index)

        probe_loader = _build_probe_loader(
            full_loader=train_loader,
            probe_fraction=probe_fraction,
            min_samples=min_probe_samples,
            max_samples=max_probe_samples,
            seed=base_seed + client_index,
        )

        gradient, num_samples, sum_squared_losses = _compute_client_probe_stats(
            learner=reference_learner,
            probe_loader=probe_loader,
        )

        mean_squared_loss = sum_squared_losses / float(num_samples)
        sigma_value = float(num_samples) / math.sqrt(max(mean_squared_loss, 1e-12))

        stats_list.append(
            ClientStats(
                client_id=client_id,
                gradient=gradient.detach().cpu(),
                num_probe_samples=num_samples,
                sum_squared_losses=sum_squared_losses,
                sigma=sigma_value,
            )
        )
        gradients.append(gradient.to(device))

    if not gradients:
        raise RuntimeError("No gradients computed; all clients have empty train iterators.")

    gradient_matrix = torch.stack(gradients, dim=0)
    distance_matrix = _pairwise_l2_distances(gradient_matrix).cpu()
    sigma_vector = torch.tensor([stats.sigma for stats in stats_list], dtype=torch.float32)

    return distance_matrix, sigma_vector, stats_list
