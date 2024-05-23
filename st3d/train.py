"""File containing training script
"""
# ======== standard imports ========
import os
import multiprocessing as mp
import threading
import queue
# ==================================

# ======= third party imports ======
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# ==================================

# ========= program imports ========
from st3d.models import StudentTeacher, Decoder
import st3d.consts as consts
import st3d.analytical as anly
# ==================================

# ==================== DATA ====================

def get_all_pretraining_data():
    np_arrs = []
    for fpath in sorted(os.listdir('point_clouds')):
        np_arrs.append(np.load(os.path.join('point_clouds', fpath)))
    training_data = torch.from_numpy(np.array(np_arrs, dtype=np.float32)[:consts.NUM_TRAINING_SCENES])
    validation_data = torch.from_numpy(np.array(np_arrs, dtype=np.float32)[-consts.NUM_VALIDATION_SCENES:]) # Just don't make this 0
    return training_data, validation_data

def get_mvtec_data():
    pass

# ==================== PREPROCESSING ====================

def precompute_geometric_properties(
        point_cloud: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        geom_features, closest_indices = anly.calculate_geom_features(point_cloud.to(consts.DEVICE), consts.K)
        #receptive_fields = anly.get_receptive_fields(closest_indices)
        return geom_features.to('cpu'), closest_indices.to('cpu')

def generate_model_pass_iterable(point_cloud_data:torch.Tensor):
    # Data is ?xNx3
    iterable = []
    for point_cloud in tqdm(point_cloud_data):
        data_item = (point_cloud, *precompute_geometric_properties(point_cloud))
        iterable.append(data_item)
    return iterable

def get_avg_distance_scalar(train_data:torch.Tensor):
    # Data is ?xNx3
    summed_norms = 0
    for point_cloud in tqdm(train_data):
        _, norms = anly.calc_point2point_diffs(point_cloud.unsqueeze(0))
        summed_norms += norms.sum()
    return summed_norms/(consts.K*consts.N)

# ==================== LOSSES ====================
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

# ==================== MODEL PASSES ====================
def td_model_pass(
        teacher:StudentTeacher, 
        decoder:Decoder, 
        point_cloud:torch.Tensor,
        geom_features:torch.Tensor, 
        closest_indices:torch.Tensor,
    ):
    point_cloud = point_cloud.to(consts.DEVICE)
    geom_features = geom_features.to(consts.DEVICE)
    closest_indices = closest_indices.to(consts.DEVICE)
    point_features = teacher(point_cloud, geom_features, closest_indices)

    chosen_idxs = torch.randint(0, consts.N, (consts.DECODER_SAMPLE_SIZE,))
    receptive_fields = anly.get_receptive_fields(point_cloud.cpu(), closest_indices.cpu(), chosen_idxs)
    receptive_field_diffs = [
        (receptive_field - torch.mean(receptive_field, dim=0)).to(consts.DEVICE)
        for receptive_field in receptive_fields
    ]

    chosen_point_features = point_features[chosen_idxs]
    decoded_points = decoder(chosen_point_features)
    
    loss = torch.zeros(1, device=consts.DEVICE)
    for i in range(len(chosen_idxs)):
        loss += chamfer_distance(decoded_points[i], receptive_field_diffs[i])
    loss /= len(chosen_idxs)

    return loss

# ==================== TRAINING LOOPS ====================
def train_td(
        train_dset:torch.Tensor, 
        val_dset:torch.Tensor, 
        teacher:StudentTeacher, 
        decoder:Decoder, 
        optimizer:torch.optim.Adam
    ):
    pbar = tqdm(range(consts.NUM_PRETRAINING_EPOCHS), position=0, leave=True)
    all_training_losses = []
    all_validation_losses = []
    for epoch in pbar:
        training_losses = []
        validation_losses = []
        np.random.shuffle(train_dset)

        for items in train_dset:
            optimizer.zero_grad()
            loss = td_model_pass(teacher, decoder, *items)
            training_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for items in val_dset:
                loss = td_model_pass(teacher, decoder, *items)
                validation_losses.append(loss.item())
                
        pbar.set_postfix({'train_loss': np.mean(training_losses), "val_loss":np.mean(validation_losses)})

        all_training_losses.append(np.array(training_losses))
        all_validation_losses.append(np.array(validation_losses))
    np.save('pre_training_losses.npy', np.array(all_training_losses))
    np.save('pre_validation_losses.npy', np.array(all_validation_losses))
    torch.save(teacher.state_dict(), f"pretrained_teacher.pth")
    torch.save(decoder.state_dict(), f"pretrained_teacher.pth")

# ==================== TRAINING SETUP ====================

def pretrain():
    train_data, val_data = get_all_pretraining_data()
    print('Loaded in point clouds from memory')

    normalization_s = get_avg_distance_scalar(train_data)
    normalization_s = 17.4138
    train_data/=normalization_s
    val_data/=normalization_s
    print('Calculated normalization constant (s): as: ' + str(normalization_s))

    print('Beginnning precomputation of geometric properties')
    train_dset = generate_model_pass_iterable(train_data)
    val_dset = generate_model_pass_iterable(val_data)
    print('Precomputed Geometric Properties for training/validation iterables')

    teacher = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS).to(device=consts.DEVICE)
    decoder = Decoder(consts.D).to(consts.DEVICE)
    optimizer = torch.optim.Adam(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr = consts.PRETRAINING_LR,
        weight_decay=consts.PRETRAINING_WD
    )
    train_td(train_dset, val_dset, teacher, decoder, optimizer)
    return teacher, normalization_s

def train_student(teacher, normalization_s):
    train_data, val_data = get_mvtec_data()
    print('Loaded in point clouds from memory')

    train_data/=normalization_s
    val_data/=normalization_s
    print('Normalized data with normalization constant (s): as: ' + str(normalization_s))

    student = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS).to(device=consts.DEVICE)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr = consts.PRETRAINING_LR,
        weight_decay=consts.PRETRAINING_WD
    )






    


if __name__ == "__main__":
    pretrain()