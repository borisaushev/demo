import cv2
import numpy as np  
import open3d as o3d
import time

from utils.constants import *
from utils.paths import *
from utils.pcd_proccessing import *
from utils.visuals import *

class FloorFinder:
    def __init__(self, pcd):
        self.pcd = pcd
    
    def get_floor_grid(self):
        pcd = self.pcd
        downsampled_pcd = downsample(pcd)

        #Ищем плоскость пола и точки в ней и вне неё
        plane_model, _ = downsampled_pcd.segment_plane(distance_threshold = distance_threshold,
                                                    num_iterations = ransac_iterations,
                                                    ransac_n = 3)
        [a, b, c, d] = plane_model
        #строим нормаль к поверхности   
        normal = np.array([a, b, c])

        # Убираем слишком далекие от пола точки
        pcd_points = np.asarray(pcd.points)
        close_mask = np.abs(a * pcd_points[:, 0] 
                        + b * pcd_points[:, 1] 
                        + c * pcd_points[:, 2] 
                        + d) <= max_dist
        close_points = pcd_points[close_mask]

        close_points_ds = voxel_downsample_fast(close_points, grid_step)

        # Разделяем на inliers и outliers
        close_distances_ds = np.abs(np.dot(close_points_ds, normal) + d)
        inlier_mask = close_distances_ds <= distance_threshold
        inliers = close_points_ds[inlier_mask]
        outliers = close_points_ds[~inlier_mask]

        #меняем направление нормали в сторону большего количества точек
        condition = np.dot(outliers, normal) + d < 0
        if len(outliers[condition]) > len(outliers) // 2:
            normal = -normal

        # берем 2 перпендикулярных вектора в плоскости
        v1_norm = np.linalg.norm([b, -a, 0])
        plane_v1 = np.asarray([b, -a, 0]) / v1_norm
        plane_v2 = np.cross(normal, plane_v1)

        #проецируем точки пола на плоскость пола
        projected = project_to_plane(inliers, plane_model)

        #переходим от координат в пространстве в координаты 2-х векторов в плоскости
        plane_coords, x_max, x_min, y_max, y_min = switch_to_plane_coords(projected, plane_v1, plane_v2)

        #строим двумерный массив, соответствующий плоскости
        grid = np.zeros((
            int((y_max - y_min) // grid_step + 1), 
            int((x_max - x_min) // grid_step + 1)
        ))
        # Инвертируем Y т.к. в opencv 0 по оси Y это самый верх
        x_indices = ((plane_coords[:, 0] - x_min) // grid_step).astype(int)
        y_indices = ((y_max - plane_coords[:, 1]) // grid_step).astype(int) 
        valid_mask = (x_indices >= 0) & (x_indices < grid.shape[1]) & \
                    (y_indices >= 0) & (y_indices < grid.shape[0])

        grid[y_indices[valid_mask], x_indices[valid_mask]] = 1

        # наращиваем точки пола
        kernel = np.ones((morph_fill_size,morph_fill_size))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)


        # проецируем точки вне пола на плоскость
        projected_outliers = project_to_plane(outliers, plane_model)

        # убираем из плоскости точки, где есть препятствия
        outliers_plane_coords = switch_to_plane_coords(projected_outliers, plane_v1, plane_v2)[0]
        x_indices = ((outliers_plane_coords[:, 0] - x_min) // grid_step).astype(int)
        y_indices = ((y_max - outliers_plane_coords[:, 1]) // grid_step).astype(int)
        valid_mask = (x_indices >= 0) & (x_indices < grid.shape[1]) & \
                    (y_indices >= 0) & (y_indices < grid.shape[0])

        floor_obstacles = grid
        floor_obstacles[y_indices[valid_mask], x_indices[valid_mask]] = 0

        kernel = np.ones((morph_size,morph_size))
        floor_obstacles = cv2.morphologyEx(floor_obstacles, cv2.MORPH_OPEN, kernel)

        #убираем маленькие компоненты 
        components_info = cv2.connectedComponentsWithStats(
            floor_obstacles.astype(np.uint8), 
            connectivity=4, 
            ltype=cv2.CV_32S
        )
        [num_labels, labels, stats, centroids] = components_info
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]*grid_step**2
            if area < min_area:
                floor_obstacles[labels == i] = 0

        # сначала жестко так прям убираем шум
        kernel = np.ones((morph_fill_size, morph_fill_size))
        floor_polished = cv2.morphologyEx(floor_obstacles, cv2.MORPH_CLOSE, kernel)

        # теперь возвращаем препятствия, которые стерлись как шум
        floor_accessible_reversed = 1 - floor_obstacles
        floor_polished_reversed = 1 - floor_polished
        components_info = cv2.connectedComponentsWithStats(
            floor_accessible_reversed.astype(np.uint8), 
            connectivity=4, 
            ltype=cv2.CV_32S
        )
        [num_labels, labels, stats, centroids] = components_info
        for i in range(1, num_labels):
            obstacle = floor_accessible_reversed[labels == i]
            if np.any(cv2.bitwise_and(obstacle, floor_polished_reversed[labels == i])):
                floor_polished[labels == i] = 0

        return floor_polished