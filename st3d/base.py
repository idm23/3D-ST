"""File containing high level functions for training and testing 3D-ST
"""
# ======== standard imports ========
import os
# ==================================

# ======= third party imports ======
import matplotlib.pyplot as plt
import numpy as np
# ==================================

# ========= program imports ========
from st3d.synthetic_data import generate_synthetic_data
# ==================================

def main():
    generate_synthetic_data()

if __name__ == "__main__":
    main()
