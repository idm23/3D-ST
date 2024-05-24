"""File containing training script for Student
"""
# ======== standard imports ========
import os
from pathlib import Path
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
import st3d.training_scripts.common as common
# ==================================

def calculate_fmap_distribution(t_fmaps:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Shape (len(training_set)xNxd)
    rolled_fmaps = t_fmaps.reshape(-1, t_fmaps.shape[-1])
    std, mu = torch.std_mean(rolled_fmaps, dim = 0, keepdim=True)
    return std, mu

def normalize_t_fmap(t_fmap:torch.Tensor, mu:torch.Tensor, std:torch.Tensor):
    pass

def model_pass(
        student:StudentTeacher,
        point_cloud:torch.Tensor,
        geom_features:torch.Tensor, 
        closest_indices:torch.Tensor,
        normalized_t_fmap:torch.Tensor
    ):
    point_cloud = point_cloud.to(consts.DEVICE)
    geom_features = geom_features.to(consts.DEVICE)
    closest_indices = closest_indices.to(consts.DEVICE)
    normalized_t_fmap = normalized_t_fmap.to(consts.DEVICE)
    s_fmap = student(point_cloud, geom_features, closest_indices)
    
    loss = ((s_fmap - normalized_t_fmap) ** 2).sum()/point_cloud.shape[0]
    

    return loss

def train(
        train_dset:torch.Tensor, 
        val_dset:torch.Tensor, 
        student:StudentTeacher, 
        optimizer:torch.optim.Adam
    ):
    pbar = tqdm(range(consts.NUM_TRAINING_EPOCHS), position=0, leave=True)
    all_training_losses = []
    all_validation_losses = []
    for epoch in pbar:
        training_losses = []
        validation_losses = []
        np.random.shuffle(train_dset)

        for items in train_dset:
            optimizer.zero_grad()
            loss = model_pass(student *items)
            training_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for items in val_dset:
                loss = model_pass(student, *items)
                validation_losses.append(loss.item())
                
        pbar.set_postfix({'train_loss': np.mean(training_losses), "val_loss":np.mean(validation_losses)})

        all_training_losses.append(np.array(training_losses))
        all_validation_losses.append(np.array(validation_losses))
    np.save('pre_training_losses.npy', np.array(all_training_losses))
    np.save('pre_validation_losses.npy', np.array(all_validation_losses))
    torch.save(teacher.state_dict(), "pretrained_teacher.pth")

def regress(
        teacher, 
        normalization_s,
        data_dir,
        save_dir
    ):
    data_dir = os.path.join(*(*this_fpath, data_dir))
    save_dir = os.path.join(*(*this_fpath, save_dir))

    train_data, val_data = common.get_all_data_from_memory(data_dir, 10, 5)
    print('Loaded in point clouds from memory')

    train_data/=normalization_s
    val_data/=normalization_s
    print('Normalized data with normalization constant (s): as: ' + str(normalization_s))

    print('Beginnning precomputation of geometric properties')
    train_dset = common.generate_model_pass_iterable(train_data)
    val_dset = common.generate_model_pass_iterable(val_data)
    print('Precomputed Geometric Properties for training/validation iterables')

    print('Precomputing Teacher feature maps')
    with torch.no_grad():
        train_tfmaps = [
            teacher(
                point_cloud.to(consts.DEVICE), 
                geom_features.to(consts.DEVICE), 
                closest_indices.to(consts.DEVICE)
            ).to('cpu')
            for (point_cloud, geom_features, closest_indices) in tqdm(train_dset)
        ]
        val_tfmaps = [
            teacher(
                point_cloud.to(consts.DEVICE), 
                geom_features.to(consts.DEVICE), 
                closest_indices.to(consts.DEVICE)
            ).to('cpu')
            for (point_cloud, geom_features, closest_indices) in tqdm(val_dset)
        ]
    train_tfmaps = torch.vstack(train_tfmaps)
    val_tfmaps = torch.vstack(val_tfmaps)
    print(f'Precomputed Teacher feature maps with train shape {train_tfmaps.shape} and val shape {val_tfmaps.shape}')

    print('Normalizing Teacher feature maps')
    mu, sigma = calculate_fmap_distribution(train_tfmaps)
    train_tfmaps = (train_tfmaps - mu)/sigma
    val_tfmaps = (train_tfmaps - mu)/sigma
    print(f'Normalized Teacher feature maps with mu: {mu} and sigma: {sigma}')

    train_dset = [
        (point_cloud, geom_features, closest_indices, train_tfmap) 
        for (point_cloud, geom_features, closest_indices), train_tfmap 
        in zip(train_dset, train_tfmaps)
    ]
    val_dset = [
        (point_cloud, geom_features, closest_indices, val_tfmap) 
        for (point_cloud, geom_features, closest_indices), val_tfmap in 
        zip(val_dset, val_tfmaps)
    ]

    student = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS).to(device=consts.DEVICE)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr = consts.PRETRAINING_LR,
        weight_decay=consts.PRETRAINING_WD
    )

if __name__ == "__main__":
    print('Using device: ' +str(consts.DEVICE))
    this_fpath = os.path.split(Path(__file__).absolute())[:-1]
    teacher_path = os.path.join(*(*this_fpath, "../training_saves/pretrained_teacher.pth"))
    teacher = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS).to(consts.DEVICE)
    teacher.load_state_dict(torch.load(teacher_path))
    normalization_s = 17.4138
    regress(teacher, normalization_s, '../real_point_clouds', '../training_saves')