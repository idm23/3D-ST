"""File containing functions for generating real dataset
"""
# ======== standard imports ========
import os
from pathlib import Path
import json
# ==================================

# ======= third party imports ======
import torch
import numpy as np
from tqdm import tqdm
from tifffile import tifffile
# ==================================

# ========= program imports ========
import st3d.consts as consts
import st3d.visualize as viz
# ==================================

def farthest_point_sampling(
        points:torch.Tensor, 
        num_points_in_cloud:int
    ) -> torch.Tensor:
    N, _ = points.shape
    sampled_indices = torch.zeros(num_points_in_cloud, dtype=torch.long)
    distances = torch.full((N,), float('inf'), device=points.device)
    
    # Initialize the first point randomly
    sampled_indices[0] = torch.randint(0, N, (1,), device=points.device)
    farthest_point = points[sampled_indices[0]].unsqueeze(0)
    
    for i in range(1, num_points_in_cloud):
        # Update distances with the minimum distance to the current farthest point
        dist_to_farthest_point = torch.norm(points - farthest_point, dim=1)
        distances = torch.min(distances, dist_to_farthest_point)
        
        # Select the next farthest point
        sampled_indices[i] = torch.argmax(distances)
        farthest_point = points[sampled_indices[i]].unsqueeze(0)
    
    return points[sampled_indices]

def rectify_xyz(arr:torch.Tensor, camera_parameters:dict) -> torch.Tensor:
    cx = camera_parameters["cx"]
    cy = camera_parameters["cy"]
    sx = camera_parameters["sx"]
    sy = camera_parameters["sy"]
    focus = camera_parameters["focus"]
    kappa = camera_parameters["kappa"]

    # Convert from pixel coordinates to normalized image coordinates
    x = (arr[:, 0] - cx) * sx
    y = (arr[:, 1] - cy) * sy
    z = arr[:, 2] * focus
    
    r_squared = x**2 + y**2
    factor = 1 + kappa * r_squared
    
    x_corrected = x * factor
    y_corrected = y * factor
    
    # Create transformed point cloud
    transformed_points = torch.stack((x_corrected, y_corrected, z), dim=1)
    transformed_points = transformed_points[transformed_points[:, 2] != 0]
    return transformed_points

def construct_point_cloud(arr, camera_parameters, num_points_in_cloud):
    arr = torch.from_numpy(arr.reshape(-1, 3).astype(np.float32)).to(consts.DEVICE)
    arr = rectify_xyz(arr, camera_parameters)
    if arr.shape[0] > num_points_in_cloud:
        point_cloud = farthest_point_sampling(arr, num_points_in_cloud)
    elif arr.shape[0] < num_points_in_cloud:
        arr = torch.stack((arr, arr[:num_points_in_cloud-arr.shape[0]]))
        point_cloud = arr
    else:
        point_cloud = arr
    point_cloud = point_cloud.cpu().numpy()
    return point_cloud
    #viz.visualize_point_cloud(point_cloud)

def build_point_cloud_from_fpath(
        num_points_in_cloud:int, 
        fpath:str, 
        camera_parameters:dict,
        output_id:int, 
        output_dir:str
    ) -> None:
    output_filename = os.path.join(output_dir, f'point_cloud_{output_id}.npy')
    if os.path.exists(output_filename):
        pass
    else:
        arr = tifffile.imread(fpath)
        point_cloud = construct_point_cloud(arr, camera_parameters, num_points_in_cloud)
        np.save(output_filename, point_cloud)

def generate_mvtec_pointclouds(
        num_points_in_cloud: int = consts.NUM_POINTS_IN_CLOUD, 
        data_dir:str = '../../MVTEC', 
        output_dir:str = '../real_point_clouds'
    ) -> None:
    
    this_fpath = os.path.split(Path(__file__).absolute())[:-1]
    data_dir = os.path.join(*(*this_fpath, data_dir))
    output_dir = os.path.join(*(*this_fpath, output_dir))
    assert os.path.exists(data_dir)

    os.makedirs(output_dir, exist_ok=True)
    model_types = sorted(os.listdir(data_dir))
    all_fpaths = []
    all_camera_parameters = []
    for model_type in model_types:
        with open(os.path.join(data_dir, model_type, 'calibration', 'camera_parameters.json'), 'r') as f:
            camera_parameters = json.load(f)
        basic_root = os.path.join(data_dir, model_type, 'train', 'good', 'xyz')
        for fname in sorted(os.listdir(basic_root)):
            all_fpaths.append(os.path.join(basic_root, fname))
            all_camera_parameters.append(camera_parameters)

    for items in tqdm([
        (num_points_in_cloud, all_fpaths[i], all_camera_parameters[i], i, output_dir) 
        for i in range(len(all_fpaths))
    ]):
        build_point_cloud_from_fpath(*items)

if __name__ == '__main__':
    generate_mvtec_pointclouds()