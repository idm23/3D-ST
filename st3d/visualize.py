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
from tifffile import tifffile
# ==================================

def visualize_point_cloud(point_cloud:np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()

def visualize_training_losses(training_losses, validation_losses) -> None:
    x = np.arange(consts.NUM_PRETRAINING_EPOCHS)
    plt.plot(x, np.mean(training_losses, axis=1), label = 'Train')
    plt.plot(x, np.mean(validation_losses, axis=1), label = 'Test')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    this_fpath = os.path.split(Path(__file__).absolute())[:-1]
    visualize_point_cloud(np.load(os.path.join(*(*this_fpath, 'syn_point_clouds/point_cloud_0.npy'))))
    visualize_point_cloud(np.load(os.path.join(*(*this_fpath, 'real_point_clouds/point_cloud_500.npy'))))
    visualize_training_losses(
        np.load(os.path.join(*(*this_fpath, 'training_saves/pre_training_losses.npy'))),
        np.load(os.path.join(*(*this_fpath, 'training_saves/pre_validation_losses.npy')))
    )

