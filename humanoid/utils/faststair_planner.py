"""GPU-parallel DCM foothold search for the N2 FastStair curriculum.

The planner is intentionally independent of Isaac Gym.  The environment
constructs terrain-aware candidate footholds, while this module scores every
candidate with vectorized Torch operations.  Keeping the dynamics/search
kernel pure makes it possible to validate symmetry, fallback behaviour, and
numerical stability on a CPU-only development machine.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DCMPlannerWeights:
    """Dimensionless weights and safety limits for foothold search."""

    nominal: float = 1.0
    dcm_offset: float = 2.0
    steepness: float = 1.0
    edge: float = 0.35
    gravity: float = 9.81
    min_com_height: float = 0.30
    max_com_height: float = 1.20
    min_horizon: float = 0.12
    max_horizon: float = 0.45
    max_growth_exponent: float = 2.0
    nominal_scale_x: float = 0.15
    nominal_scale_y: float = 0.10
    dcm_scale_x: float = 0.20
    dcm_scale_y: float = 0.15


@dataclass
class DCMPlannerResult:
    """Selected footholds and diagnostics for a batch of environments."""

    foothold: torch.Tensor
    cost: torch.Tensor
    valid: torch.Tensor
    selected_index: torch.Tensor
    dcm: torch.Tensor
    predicted_dcm: torch.Tensor
    dcm_offset: torch.Tensor


def rectangular_search_offsets(
    x_offsets,
    y_offsets,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return a deterministic ``(len(x) * len(y), 2)`` search grid."""
    x = torch.as_tensor(x_offsets, device=device, dtype=dtype)
    y = torch.as_tensor(y_offsets, device=device, dtype=dtype)
    if x.ndim != 1 or y.ndim != 1 or x.numel() == 0 or y.numel() == 0:
        raise ValueError("FastStair search offsets must be non-empty vectors")
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)


def _require_shape(name: str, value: torch.Tensor, trailing_shape) -> None:
    if value.ndim != 1 + len(trailing_shape):
        raise ValueError(
            "{} must have {} dimensions, received {}".format(
                name, 1 + len(trailing_shape), tuple(value.shape)
            )
        )
    if tuple(value.shape[1:]) != tuple(trailing_shape):
        raise ValueError(
            "{} must end in {}, received {}".format(
                name, tuple(trailing_shape), tuple(value.shape)
            )
        )


def dcm_foothold_search(
    *,
    com_xy: torch.Tensor,
    com_velocity_xy: torch.Tensor,
    com_height: torch.Tensor,
    stance_xy: torch.Tensor,
    desired_velocity_xy: torch.Tensor,
    nominal_foothold: torch.Tensor,
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    horizon: torch.Tensor,
    candidate_steepness: Optional[torch.Tensor] = None,
    candidate_edge_cost: Optional[torch.Tensor] = None,
    weights: DCMPlannerWeights = DCMPlannerWeights(),
) -> DCMPlannerResult:
    """Select dynamically feasible footholds with a parallel DCM search.

    All planar positions are expressed in one common world frame.  Candidate
    Z values are carried through selection but do not enter the planar DCM
    model.  When no terrain candidate is valid, the nominal foothold is used
    exactly and ``result.valid`` is false, allowing the caller to suppress
    planner rewards without producing NaNs.
    """
    _require_shape("com_xy", com_xy, (2,))
    _require_shape("com_velocity_xy", com_velocity_xy, (2,))
    _require_shape("stance_xy", stance_xy, (2,))
    _require_shape("desired_velocity_xy", desired_velocity_xy, (2,))
    _require_shape("nominal_foothold", nominal_foothold, (3,))
    if candidates.ndim != 3 or candidates.shape[2] != 3:
        raise ValueError(
            "candidates must have shape (batch, candidates, 3), received "
            + str(tuple(candidates.shape))
        )
    batch_size, candidate_count, _ = candidates.shape
    batch_tensors = (
        com_xy,
        com_velocity_xy,
        com_height,
        stance_xy,
        desired_velocity_xy,
        nominal_foothold,
        candidate_valid,
        horizon,
    )
    if any(value.shape[0] != batch_size for value in batch_tensors):
        raise ValueError("FastStair planner inputs have inconsistent batches")
    if candidate_valid.shape != (batch_size, candidate_count):
        raise ValueError(
            "candidate_valid must have shape {}, received {}".format(
                (batch_size, candidate_count),
                tuple(candidate_valid.shape),
            )
        )
    for name, value in (
        ("candidate_steepness", candidate_steepness),
        ("candidate_edge_cost", candidate_edge_cost),
    ):
        if value is not None and value.shape != (
            batch_size,
            candidate_count,
        ):
            raise ValueError(
                "{} must have shape {}, received {}".format(
                    name,
                    (batch_size, candidate_count),
                    tuple(value.shape),
                )
            )

    dtype = candidates.dtype
    device = candidates.device
    com_height = torch.clamp(
        com_height.to(dtype=dtype),
        min=float(weights.min_com_height),
        max=float(weights.max_com_height),
    )
    horizon = torch.clamp(
        horizon.to(dtype=dtype),
        min=float(weights.min_horizon),
        max=float(weights.max_horizon),
    )
    omega = torch.sqrt(
        torch.full_like(com_height, float(weights.gravity)) / com_height
    )
    dcm = com_xy + com_velocity_xy / omega.unsqueeze(1)
    exponent = torch.clamp(
        omega * horizon,
        min=0.0,
        max=float(weights.max_growth_exponent),
    )
    predicted_dcm = stance_xy + (
        dcm - stance_xy
    ) * torch.exp(exponent).unsqueeze(1)

    candidate_xy = candidates[:, :, :2]
    candidate_dcm_offset = predicted_dcm.unsqueeze(1) - candidate_xy
    desired_dcm_offset = (
        desired_velocity_xy / omega.unsqueeze(1)
    ).unsqueeze(1)

    nominal_scale = torch.tensor(
        (weights.nominal_scale_x, weights.nominal_scale_y),
        device=device,
        dtype=dtype,
    ).clamp_min(1.0e-4)
    dcm_scale = torch.tensor(
        (weights.dcm_scale_x, weights.dcm_scale_y),
        device=device,
        dtype=dtype,
    ).clamp_min(1.0e-4)
    nominal_error = (
        candidate_xy - nominal_foothold[:, None, :2]
    ) / nominal_scale
    dcm_error = (
        candidate_dcm_offset - desired_dcm_offset
    ) / dcm_scale
    cost = (
        float(weights.nominal)
        * torch.sum(torch.square(nominal_error), dim=2)
        + float(weights.dcm_offset)
        * torch.sum(torch.square(dcm_error), dim=2)
    )
    if candidate_steepness is not None:
        cost = cost + float(weights.steepness) * torch.square(
            candidate_steepness.to(dtype=dtype)
        )
    if candidate_edge_cost is not None:
        cost = cost + float(weights.edge) * candidate_edge_cost.to(
            dtype=dtype
        )

    finite_candidate = torch.isfinite(candidates).all(dim=2)
    finite_cost = torch.isfinite(cost)
    valid_mask = candidate_valid.bool() & finite_candidate & finite_cost
    large_cost = torch.finfo(dtype).max / 16.0
    masked_cost = torch.where(
        valid_mask,
        cost,
        torch.full_like(cost, large_cost),
    )
    selected_index = torch.argmin(masked_cost, dim=1)
    gather_xyz = selected_index.view(-1, 1, 1).expand(-1, 1, 3)
    selected = torch.gather(candidates, 1, gather_xyz).squeeze(1)
    gather_scalar = selected_index.unsqueeze(1)
    selected_cost = torch.gather(
        masked_cost, 1, gather_scalar
    ).squeeze(1)
    selected_offset = torch.gather(
        candidate_dcm_offset,
        1,
        selected_index.view(-1, 1, 1).expand(-1, 1, 2),
    ).squeeze(1)
    any_valid = torch.any(valid_mask, dim=1)
    selected = torch.where(
        any_valid.unsqueeze(1), selected, nominal_foothold
    )
    selected_cost = torch.where(
        any_valid, selected_cost, torch.zeros_like(selected_cost)
    )
    fallback_offset = predicted_dcm - nominal_foothold[:, :2]
    selected_offset = torch.where(
        any_valid.unsqueeze(1), selected_offset, fallback_offset
    )
    return DCMPlannerResult(
        foothold=selected,
        cost=selected_cost,
        valid=any_valid,
        selected_index=selected_index,
        dcm=dcm,
        predicted_dcm=predicted_dcm,
        dcm_offset=selected_offset,
    )
