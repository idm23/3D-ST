"""File containing visualization functions
"""
# ======== standard imports ========
import os
from pathlib import Path
# ==================================

# ======= third party imports ======
import torch
import numpy as np
import matplotlib.pyplot as plt
# ==================================

# ========= program imports ========
from st3d.models import StudentTeacher, Decoder
import st3d.consts as consts
import st3d.analytical as anly
import st3d.training_scripts.common as common
# ==================================

def visualize_point_cloud(point_cloud:np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()

def visualize_receptive_field(point_cloud:np.ndarray) -> None:
    points = torch.from_numpy(point_cloud)
    _, closest_indices = anly.calculate_geom_features(points, consts.K)
    receptive_field, receptive_field_idxs = anly.get_receptive_fields(points, closest_indices, torch.arange(16))
    receptive_field = receptive_field[4].numpy()
    receptive_field_idxs = receptive_field_idxs[4].numpy()
    print(receptive_field.shape)

    # Create a mask for the points in the receptive field
    mask = np.zeros(point_cloud.shape[0], dtype=bool)
    mask[receptive_field_idxs] = True
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(point_cloud[~mask, 0], point_cloud[~mask, 1], point_cloud[~mask, 2], marker='o', c='blue', label='Remaining Points')
    ax.scatter(point_cloud[mask, 0], point_cloud[mask, 1], point_cloud[mask, 2], marker='^', c='orange', label='Receptive Field')
    plt.show()

def visualize_training_losses(training_losses, validation_losses) -> None:
    x = np.arange(training_losses.shape[0])
    plt.plot(x, np.mean(training_losses, axis=1), label = 'Train')
    plt.plot(x, np.mean(validation_losses, axis=1), label = 'Test')
    plt.legend()
    plt.show()

def visualize_anomaly_prediction(
        teacher_path, 
        student_path, 
        anomalous_data_path, 
        mu_path,
        sigma_path,
        normalization_s
    ):
    teacher = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS)
    student = StudentTeacher(consts.K, consts.D, consts.NUM_RESIDUALS)
    teacher = teacher.to(consts.DEVICE)
    student = student.to(consts.DEVICE)
    teacher.load_state_dict(torch.load(teacher_path))
    student.load_state_dict(torch.load(student_path))
    teacher.eval()
    student.eval()
    point_cloud = np.load(anomalous_data_path)
    points = torch.from_numpy(point_cloud).to(consts.DEVICE)/normalization_s
    mu = torch.from_numpy(np.load(mu_path)).to(consts.DEVICE)
    sigma = torch.from_numpy(np.load(sigma_path)).to(consts.DEVICE)
    with torch.no_grad():
        teacher_fmap = teacher.point_based_forward(points)
        normalized_t_fmap = (teacher_fmap - mu)/sigma
        student_fmap = student.point_based_forward(points)
        anomaly_scores = common.anomaly_score(normalized_t_fmap, student_fmap).cpu().numpy()

    # Normalize anomaly_scores
    normalized_scores = anomaly_scores / np.max(anomaly_scores)

    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=normalized_scores, cmap='viridis')

    # Add color bar to indicate the score scale
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Score')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    this_fpath = os.path.split(Path(__file__).absolute())[:-1]
    """visualize_receptive_field(np.load(os.path.join(*(*this_fpath, 'syn_point_clouds/point_cloud_0.npy'))))
    visualize_point_cloud(np.load(os.path.join(*(*this_fpath, 'real_point_clouds/point_cloud_0.npy'))))
    visualize_training_losses(
        np.load(os.path.join(*(*this_fpath, 'training_saves/pre_training_losses.npy'))),
        np.load(os.path.join(*(*this_fpath, 'training_saves/pre_validation_losses.npy')))
    )
    visualize_training_losses(
        np.load(os.path.join(*(*this_fpath, 'training_saves/training_losses.npy'))),
        np.load(os.path.join(*(*this_fpath, 'training_saves/validation_losses.npy')))
    )"""
    visualize_anomaly_prediction(
        os.path.join(*(*this_fpath, 'training_saves/pretrained_teacher.pth')),
        os.path.join(*(*this_fpath, 'training_saves/trained_student.pth')),
        os.path.join(*(*this_fpath, 'defective_point_clouds/point_cloud_0.npy')),
        os.path.join(*(*this_fpath, 'training_saves/mu.npy')),
        os.path.join(*(*this_fpath, 'training_saves/sigma.npy')),
        2.4867298576282336e-07
    )

