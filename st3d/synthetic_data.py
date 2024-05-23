"""File containing functions for generating synthetic dataset
"""
# ======== standard imports ========
import os
import multiprocessing as mp
# ==================================

# ======= third party imports ======
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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
    return np.array(verts), np.array(faces)

def get_all_fpaths(data_path: str) -> list[str]:
    all_obj_type_paths = os.listdir(data_path)
    all_training_fpaths = []
    for obj_type_path in all_obj_type_paths:
        cpath = os.path.join(data_path, obj_type_path, 'train')
        all_training_fpaths += [os.path.join(cpath, fname) for fname in os.listdir(cpath)]
    return all_training_fpaths

def scale_verts(verts: np.ndarray) -> np.ndarray:
    verts /= np.max(np.abs(verts))
    return verts

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
    R = Rz @ Ry @ Rx
    return verts @ R.T

def translate_verts(verts: np.ndarray) -> np.ndarray:
    translation = np.random.uniform(-3, 3, 3)
    return verts + translation

def transform_obj(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts = scale_verts(verts)
    verts = rotate_verts(verts)
    verts = translate_verts(verts)
    return verts, faces

def create_scene_subsets(num_scenes: int, num_objs_in_scene: int, all_fpaths: list[str]) -> list[list[str]]:
    scene_obj_paths = []
    for _ in range(num_scenes):
        cur_scene_obj_paths = []
        for _ in range(num_objs_in_scene):
            cur_scene_obj_paths.append(all_fpaths[np.random.randint(len(all_fpaths))])
        scene_obj_paths.append(cur_scene_obj_paths)
    return scene_obj_paths

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
    return all_verts, all_faces

def sample_point_from_face(vertices, face):
    v0, v1, v2 = vertices[face]
    r1, r2 = np.random.rand(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2
    return point

def farthest_point_sampling_faces(verts: np.ndarray, faces: np.ndarray, num_samples: int) -> np.ndarray:
    sampled_points = np.zeros((num_samples, 3))
    
    distances = np.ones(len(verts)) * 1e10
    # Sample the first point randomly from a face
    farthest = sample_point_from_face(verts, faces[np.random.choice(len(faces))])
    sampled_points[0] = farthest

    for i in range(1, num_samples):
        # Update distances from the new point
        dists = np.sum((verts - farthest) ** 2, axis=1)
        distances = np.minimum(distances, dists)
        # Select the next farthest point from all vertices
        farthest_index = np.argmax(distances)
        farthest = verts[farthest_index]

        sampled_points[i] = farthest

    return sampled_points

def save_point_cloud(point_cloud: np.ndarray, file_path: str):
    np.save(file_path, point_cloud)

def process_scene(scene_paths: list[str], num_points: int, output_dir:str, output_id:int):
    all_verts, all_faces = generate_scene(scene_paths)
    point_cloud = farthest_point_sampling_faces(all_verts, all_faces, num_points)
    output_path = os.path.join(output_dir, f"point_cloud_{output_id}.npy")
    save_point_cloud(point_cloud, output_path)

def process_scene_packed(*args):
    args = args[0]
    process_scene(*args)

def generate_synthetic_data(
        num_scenes: int = consts.NUM_TRAINING_SCENES+consts.NUM_VALIDATION_SCENES, 
        num_objs_in_scene: int = consts.NUM_OBJS_IN_SCENE,
        num_points_in_cloud: int = consts.NUM_POINTS_IN_CLOUD, 
        data_path: str = '../ModelNet10', 
        output_dir: str = './point_clouds'
    ):

    all_fpaths = get_all_fpaths(data_path)
    scene_obj_paths = create_scene_subsets(num_scenes, num_objs_in_scene, all_fpaths)
    os.makedirs(output_dir, exist_ok=True)
    
    with mp.Pool(mp.cpu_count()) as pool:
        tasks = [(scene_paths, num_points_in_cloud, output_dir, output_id) for output_id, scene_paths in enumerate(scene_obj_paths)]
        for _ in tqdm(
            pool.imap_unordered(process_scene_packed, tasks),
            position = 0, leave = True, total = len(tasks)
        ):
            pass

if __name__ == "__main__":
    generate_synthetic_data()
    