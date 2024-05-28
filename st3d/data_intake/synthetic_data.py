"""File containing functions for generating synthetic dataset
"""
# ======== standard imports ========
import os
import multiprocessing as mp
from pathlib import Path
# ==================================

# ======= third party imports ======
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import numba as nb
import torch
# ==================================

# ========= program imports ========
import st3d.consts as consts
# ==================================

def read_off(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for _ in range(n_faces)]
    return np.array(verts, dtype = np.float32), np.array(faces, dtype = np.int32)

def get_all_fpaths(data_path: str) -> list[str]:
    all_obj_type_paths = os.listdir(data_path)
    all_training_fpaths = []
    for obj_type_path in all_obj_type_paths:
        cpath = os.path.join(data_path, obj_type_path, 'train')
        all_training_fpaths += [os.path.join(cpath, fname) for fname in os.listdir(cpath)]
    return all_training_fpaths

def create_scene_subsets(num_scenes: int, num_objs_in_scene: int, all_fpaths: list[str]) -> list[list[str]]:
    scene_obj_paths = []
    for _ in range(num_scenes):
        cur_scene_obj_paths = []
        for _ in range(num_objs_in_scene):
            cur_scene_obj_paths.append(all_fpaths[np.random.randint(len(all_fpaths))])
        scene_obj_paths.append(cur_scene_obj_paths)
    return scene_obj_paths

@nb.njit
def scale_verts(verts: np.ndarray) -> np.ndarray:
    verts /= np.max(np.abs(verts))
    return verts

#@nb.njit
def rotate_verts(verts: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.random.rand(3) * 2 * np.pi
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = Ry @ Rx
    R = Rz @ R
    return np.dot(verts, R.T)

@nb.njit
def translate_verts(verts: np.ndarray) -> np.ndarray:
    translation = np.random.uniform(-3, 3, 3)
    return verts + translation

#@nb.njit
def transform_obj(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = scale_verts(verts)
    verts = rotate_verts(verts)
    verts = translate_verts(verts)
    return verts, faces

def generate_single_obj(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = read_off(obj_path)
    verts, faces = transform_obj(verts, faces)
    return verts, faces

def generate_scene(scene_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    objs = [generate_single_obj(obj_path) for obj_path in scene_paths]
    all_verts = np.vstack([obj[0] for obj in objs])
    all_faces = []
    vert_offset = 0
    for verts, faces in objs:
        all_faces.append(faces + vert_offset)
        vert_offset += verts.shape[0]
    all_faces = np.vstack(all_faces)
    return all_verts.astype(np.float32), all_faces.astype(np.int32)

@nb.njit
def triangle_area(pt1, pt2, pt3):
    side_a = np.sqrt(((pt1 - pt2)**2).sum())
    side_b = np.sqrt(((pt2 - pt3)**2).sum())
    side_c = np.sqrt(((pt3 - pt1)**2).sum())
    s = 0.5 * (side_a + side_b + side_c)
    return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

def sub_select_faces(verts, faces, num_faces):
    areas = np.zeros(len(faces))
    for i in range(len(faces)):
        areas[i] = triangle_area(verts[faces[i, 0]], verts[faces[i, 1]], verts[faces[i, 2]])
    areas /= areas.sum()

    num_sub_faces = min(num_faces, len(faces))
    sub_face_idxs = np.random.choice(len(faces), size=num_sub_faces, p=areas)
    return faces[sub_face_idxs]

def sample_point(pt1, pt2, pt3):
    s, t = sorted([torch.rand(1, device=pt1.device), torch.rand(1, device=pt1.device)])
    return s * pt1 + (t - s) * pt2 + (1 - t) * pt3

def sample_point_from_face(vertices, face):
    v0, v1, v2 = vertices[face]
    r1, r2 = torch.rand(2, device=vertices.device)
    sqrt_r1 = torch.sqrt(r1)
    point = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2
    return point

def farthest_point_sampling_faces(verts: torch.Tensor, faces: torch.Tensor, num_samples: int) -> torch.Tensor:
    sampled_points = torch.zeros((num_samples, 3), device=verts.device)
    
    face_centers = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0
    distances = torch.full((face_centers.shape[0],), float('inf'), device=verts.device)

    sampled_face_idx = np.random.randint(len(faces))
    sampled_face = faces[sampled_face_idx]
    sampled_points[0] = sample_point(verts[sampled_face[0]], verts[sampled_face[1]], verts[sampled_face[2]])

    selected_faces = [sampled_face_idx]

    for i in range(1, num_samples):
        # Update distances with the minimum distance to the current farthest point
        dist_to_farthest_point = torch.norm(face_centers - sampled_points[i-1], dim=1)
        distances = torch.min(distances, dist_to_farthest_point)
        
        # Select the next farthest point
        selected_faces.append(torch.argmax(distances))
        sampled_face = faces[selected_faces[i]]
        sampled_points[i] = sample_point(verts[sampled_face[0]], verts[sampled_face[1]], verts[sampled_face[2]])
    
    return sampled_points.cpu().numpy()

def save_point_cloud(point_cloud: np.ndarray, file_path: str):
    np.save(file_path, point_cloud)

def process_scene(scene_paths: list[str], num_points: int, output_dir:str, output_id:int):
    all_verts, all_faces = generate_scene(scene_paths)
    subset_faces = sub_select_faces(all_verts, all_faces, 3 * num_points)
    #print(len(subset_faces))
    point_cloud = farthest_point_sampling_faces(
        torch.from_numpy(all_verts).to(consts.DEVICE), 
        torch.from_numpy(subset_faces).to(consts.DEVICE), 
        num_points
    )
    output_path = os.path.join(output_dir, f"point_cloud_{output_id}.npy")
    save_point_cloud(point_cloud, output_path)

def process_scene_packed(*args):
    args = args[0]
    process_scene(*args)

def generate_synthetic_data(
        num_scenes: int = consts.NUM_TRAINING_SCENES+consts.NUM_VALIDATION_SCENES, 
        num_objs_in_scene: int = consts.NUM_OBJS_IN_SCENE,
        num_points_in_cloud: int = consts.NUM_POINTS_IN_CLOUD, 
        data_dir: str = '../../ModelNet10', 
        output_dir: str = '../syn_point_clouds'
    ):
    this_fpath = os.path.split(Path(__file__).absolute())[:-1]
    data_dir = os.path.join(*(*this_fpath, data_dir))
    output_dir = os.path.join(*(*this_fpath, output_dir))
    assert os.path.exists(data_dir)

    all_fpaths = get_all_fpaths(data_dir)
    scene_obj_paths = create_scene_subsets(num_scenes, num_objs_in_scene, all_fpaths)
    os.makedirs(output_dir, exist_ok=True)

    for task in tqdm([(scene_paths, num_points_in_cloud, output_dir, output_id+246) for output_id, scene_paths in enumerate(scene_obj_paths)]):
        process_scene(*task)
    
    """with mp.Pool(mp.cpu_count()//8) as pool:
        tasks = [(scene_paths, num_points_in_cloud, output_dir, output_id) for output_id, scene_paths in enumerate(scene_obj_paths)]
        for _ in tqdm(
            pool.imap_unordered(process_scene_packed, tasks),
            position = 0, leave = True, total = len(tasks)
        ):
            pass"""

if __name__ == "__main__":
    generate_synthetic_data()
    