"""File containing analytical functions for analyzing point cloud data
"""
# ======== standard imports ========
# ==================================

# ======= third party imports ======
import torch
import numpy as np
import numba as nb
import multiprocessing as mp
# ==================================

# ========= program imports ========
import st3d.consts as consts
# ==================================

def calc_point2point_diffs(points:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Points is Nx3
    points_expanded_1 = points.unsqueeze(1)  # Shape (N, 1, 3)
    points_expanded_2 = points.unsqueeze(0)  # Shape (1, N, 3)

    # Calculate pairwise differences
    differences = points_expanded_1 - points_expanded_2  # Shape (N, N, 3)

    norms = torch.norm(differences, dim=2) # Shape (N, N)
    return differences, norms

def get_topk_indices(norms:torch.Tensor, k:int) -> torch.Tensor:
    # Norms is NxN
    _, closest_indices = torch.topk(norms, k=k+1, largest=False)  # Shape (N, k+1)
    # Drop self
    closest_indices = closest_indices[:, 1:]  # Shape (N, k)
    return closest_indices

def calculate_geom_features(points:torch.Tensor, k:int) -> tuple[torch.Tensor, torch.Tensor]:
    # Points is Nx3
    differences, norms = calc_point2point_diffs(points)
    closest_indices = get_topk_indices(norms, k)

    closest_vector_diffs = differences[torch.arange(points.shape[0]).unsqueeze(1), closest_indices]  # Shape (N, k, 3)
    closest_scalar_diffs = norms[torch.arange(points.shape[0]).unsqueeze(1), closest_indices]  # Shape (N, k)
    
    closest_scalar_diffs = closest_scalar_diffs.unsqueeze(-1)  # Shape (N, k, 1)
    geom_features = torch.cat((closest_vector_diffs, closest_scalar_diffs), dim=-1)  # Shape (N, k, 4)

    return geom_features, closest_indices

@nb.njit
def get_rf_speed(closest_indices: np.ndarray, chosen_idxs:np.ndarray) -> list[list[int]]:
    # Shape (N, k)
    receptive_fields = []
    for chosen_idx in chosen_idxs:
        receptive_field = set([chosen_idx])
        prev_receptive_field = receptive_field
        count = 0 
        while len(receptive_field) < consts.NUM_DECODED_POINTS and count < consts.NUM_RESIDUALS * 2:
            new_receptive_field = set()
            for idx in prev_receptive_field:
                new_receptive_field.update(list(closest_indices[idx]))
            receptive_field.update(new_receptive_field)
            prev_receptive_field = new_receptive_field
            count += 1
        receptive_field = list(receptive_field)[:consts.NUM_DECODED_POINTS]
        receptive_fields.append(receptive_field)
    
    return receptive_fields

def get_receptive_fields(points:torch.Tensor, closest_indices:torch.Tensor, chosen_idxs:torch.Tensor) -> list[torch.Tensor]:
    # Shape (N, 3), (N, k), (Whatever number of samples the decoder gets)
    receptive_fields = get_rf_speed(closest_indices.numpy(), chosen_idxs.numpy())
    rf_idxs = [torch.tensor(rf, dtype=int) for rf in receptive_fields]
    receptive_fields = [points[idxs] for idxs in rf_idxs]
    return receptive_fields, rf_idxs

