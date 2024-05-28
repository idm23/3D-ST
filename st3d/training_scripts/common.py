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
    np.random.shuffle(np_arrs)
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
def chamfer_distance(point_set_gt: torch.Tensor, point_set_pred: torch.Tensor) -> torch.Tensor:
    # Compute pairwise distance matrix
    diff = point_set_gt.unsqueeze(1) - point_set_pred.unsqueeze(0)  # Shape (N, M, 3)
    sq_dist_matrix = torch.sum(diff ** 2, dim=-1) # Shape (N, M)
    
    # Grab minimum distance point to every gt point
    min_dist_1, _ = torch.min(sq_dist_matrix, dim=1)  # Shape (N)
    cdistance_1 = torch.sum(min_dist_1)/min_dist_1.shape[0]

    # Grab minimum distance point to every gt point
    min_dist_2, _ = torch.min(sq_dist_matrix, dim=0)  # Shape (M)
    cdistance_2 = torch.sum(min_dist_2)/min_dist_2.shape[0]

    return cdistance_1 + cdistance_2
    
def anomaly_score(normalized_t_fmap:torch.Tensor, s_fmap:torch.Tensor):
    return torch.norm(s_fmap - normalized_t_fmap, dim = 1)

# ==================== PREPROCESSING ====================

def get_avg_distance_scalar(train_data:torch.Tensor):
    # Data is ?xNx3
    summed_norms = 0
    for point_cloud in tqdm(train_data):
        point_cloud = point_cloud.to(consts.DEVICE)
        geom_features, _ = anly.calculate_geom_features(point_cloud, consts.K)
        
        summed_norms += geom_features[:, :, -1].sum().item()
    
    return summed_norms/(consts.K*consts.N*len(train_data))