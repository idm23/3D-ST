"""File containing functions useful for training Teacher/Decoder and Student
"""
# ======== standard imports ========
import os
# ==================================

# ======= third party imports ======
import torch
import numpy as np
from tqdm import tqdm
# ==================================

# ========= program imports ========
from st3d.models import StudentTeacher, Decoder
import st3d.consts as consts
import st3d.analytical as anly
# ==================================

# ==================== DATA ====================
def get_all_data_from_memory(
        data_dir:str,
        training_length:int,
        validation_length:int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    np_arrs = []
    for fpath in sorted(os.listdir(data_dir)):
        individual_arr = np.load(os.path.join(data_dir, fpath), allow_pickle=True)
        np_arrs.append(individual_arr)
    training_data = torch.from_numpy(np.array(np_arrs, dtype=np.float32)[:training_length])
    validation_data = torch.from_numpy(np.array(np_arrs, dtype=np.float32)[-validation_length:]) # Just don't make this 0
    return training_data, validation_data

# ==================== PREPROCESSING ====================
def precompute_geometric_properties(
        point_cloud: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geom_features, closest_indices = anly.calculate_geom_features(point_cloud.to(consts.DEVICE), consts.K)
        return geom_features.to('cpu'), closest_indices.to('cpu')

def generate_model_pass_iterable(
        point_cloud_data:torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # Data is ?xNx3
    iterable = []
    for point_cloud in tqdm(point_cloud_data):
        data_item = (point_cloud, *precompute_geometric_properties(point_cloud))
        iterable.append(data_item)
    return iterable

# ==================== LOSSES and SCORES ====================
def chamfer_distance(point_set_1: torch.Tensor, point_set_2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Chamfer distance between two sets of points.
    
    Args:
    point_set_1 (torch.Tensor): Tensor of shape (N, 3) representing the first set of points.
    point_set_2 (torch.Tensor): Tensor of shape (M, 3) representing the second set of points.
    
    Returns:
    torch.Tensor: The Chamfer distance between the two sets of points.
    """
    # Compute pairwise distance matrix
    diff = point_set_1.unsqueeze(1) - point_set_2.unsqueeze(0)  # Shape (N, M, 3)
    dist_matrix = torch.sum(diff ** 2, dim=-1)  # Shape (N, M)
    
    # Compute the nearest neighbor distances for each point in point_set_1
    min_dist_1, _ = torch.min(dist_matrix, dim=1)  # Shape (N)
    mean_dist_1 = torch.mean(min_dist_1)  # Scalar
    
    # Compute the nearest neighbor distances for each point in point_set_2
    min_dist_2, _ = torch.min(dist_matrix, dim=0)  # Shape (M)
    mean_dist_2 = torch.mean(min_dist_2)  # Scalar
    
    # Chamfer distance is the sum of the mean nearest neighbor distances
    chamfer_dist = mean_dist_1 + mean_dist_2
    
    return chamfer_dist

def anomaly_score(normalized_t_fmap:torch.Tensor, s_fmap:torch.Tensor):
    return torch.sqrt((
        (s_fmap - normalized_t_fmap) ** 2
    ).sum(dim = 1))