"""File containing training script
"""
# ======== standard imports ========
import os
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
# ==================================

def visualize_pretraining_losses():
    training_losses = np.load('pre_training_losses.npy')
    validation_losses = np.load('pre_validation_losses.npy')
    print(training_losses.shape)
    print(validation_losses.shape)
    plt.plot(np.arange(500), training_losses)
    plt.show()

if __name__ == "__main__":
    visualize_pretraining_losses()

