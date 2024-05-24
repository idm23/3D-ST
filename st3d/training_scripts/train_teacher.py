"""File containing training script for Teacher/Decoder
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

# ==================== MODEL PASSES ====================
def model_pass(
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
        loss += common.chamfer_distance(receptive_field_diffs[i], decoded_points[i])
    loss /= len(chosen_idxs)

    return loss, point_features

# ==================== TRAINING LOOPS ====================
def train(
        train_dset:torch.Tensor, 
        val_dset:torch.Tensor, 
        teacher:StudentTeacher, 
        decoder:Decoder, 
        optimizer:torch.optim.Adam,
        save_dir:str
    ):
    pbar = tqdm(range(consts.NUM_PRETRAINING_EPOCHS), position=0, leave=True)
    all_training_losses = []
    all_validation_losses = []
    for epoch in pbar:
        training_losses = []
        validation_losses = []
        np.random.shuffle(train_dset)

        teacher.train()
        decoder.train()
        all_point_features = []
        for items in train_dset:
            optimizer.zero_grad()
            loss, point_features = model_pass(teacher, decoder, *items)
            training_losses.append(loss.item())
            all_point_features.append(point_features)
            loss.backward()
            optimizer.step()

        print(torch.vstack(all_point_features).mean(dim = 0))

        teacher.eval()
        decoder.eval()
        with torch.no_grad():
            for items in val_dset:
                loss, point_features = model_pass(teacher, decoder, *items)
                validation_losses.append(loss.item())
                
        pbar.set_postfix({'train_loss': np.mean(training_losses), "val_loss":np.mean(validation_losses)})

        all_training_losses.append(np.array(training_losses))
        all_validation_losses.append(np.array(validation_losses))
    np.save(os.path.join(save_dir, 'pre_training_losses.npy'), np.array(all_training_losses))
    np.save(os.path.join(save_dir, 'pre_validation_losses.npy'), np.array(all_validation_losses))
    torch.save(teacher.state_dict(), os.path.join(save_dir, "pretrained_teacher.pth"))
    torch.save(decoder.state_dict(), os.path.join(save_dir, "pretrained_decoder.pth"))

# ==================== TRAINING SETUP ====================

def pretrain(data_dir:str, save_dir:str):
    this_fpath = os.path.split(Path(__file__).absolute())[:-1]
    data_dir = os.path.join(*(*this_fpath, data_dir))
    save_dir = os.path.join(*(*this_fpath, save_dir))

    train_data, val_data = common.get_all_data_from_memory(data_dir, consts.NUM_TRAINING_SCENES, consts.NUM_VALIDATION_SCENES)
    print('Loaded in point clouds from memory')

    print('Calculating avg distance between points')
    normalization_s = common.get_avg_distance_scalar(train_data)
    train_data/=normalization_s
    val_data/=normalization_s
    print('Calculated normalization constant (s): as: ' + str(normalization_s))

    print('Beginnning precomputation of geometric properties')
    train_dset = common.generate_model_pass_iterable(train_data)
    val_dset = common.generate_model_pass_iterable(val_data)
    print('Precomputed Geometric Properties for training/validation iterables')

    teacher = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS).to(device=consts.DEVICE)
    #teacher.load_state_dict(torch.load(os.path.join(*(*this_fpath, '../training_saves/pretrained_teacher.pth'))))
    decoder = Decoder(consts.D).to(consts.DEVICE)
    optimizer = torch.optim.Adam(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr = consts.PRETRAINING_LR,
        weight_decay=consts.PRETRAINING_WD
    )
    train(train_dset, val_dset, teacher, decoder, optimizer, save_dir)
    return teacher, normalization_s

if __name__ == "__main__":
    print('Using device: ' +str(consts.DEVICE))
    teacher, normalization_s = pretrain('../syn_point_clouds', '../training_saves')