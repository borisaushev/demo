#даунсемплинг через выбор рандомных точек
import numpy as np
from utils.constants import * 

def downsample(pcd):
    if len(pcd.points) < max_downsample_points:
        return pcd
    random_points_count = max_downsample_points
    random_indices = np.random.choice(len(pcd.points), random_points_count, replace=False)
    return pcd.select_by_index(random_indices)

# Voxel-Даунсемплинг через numpy (без Open3D)
def voxel_downsample_fast(points, voxel_size):
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    # Создаем уникальный ключ для каждого вокселя
    voxel_keys = (voxel_indices[:, 0] * 1000000 + 
                  voxel_indices[:, 1] * 1000 + 
                  voxel_indices[:, 2])
    _, unique_indices = np.unique(voxel_keys, return_index=True)
    return points[unique_indices]

#проектирование точек на плоскость
def project_to_plane(points, plane_model):
    [a, b, c, d] = plane_model
    normal = np.array([a, b, c])
    distances = np.dot(points, normal) + d

    projected_points = points - np.outer(distances, [a, b, c])
    return projected_points

#переход в систему координат плоскости (2 координаты)
def switch_to_plane_coords(projected, plane_v1, plane_v2):
    dot_v1 = np.dot(projected, plane_v1)
    dot_v2 = np.dot(projected, plane_v2)

    plane_coords = np.column_stack((dot_v1, dot_v2))
    x_max, x_min = np.max(plane_coords[:, 0]), np.min(plane_coords[:, 0])
    y_max, y_min = np.max(plane_coords[:, 1]), np.min(plane_coords[:, 1])
    return plane_coords, x_max, x_min, y_max, y_min
