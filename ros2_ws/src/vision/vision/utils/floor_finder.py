import cv2
import numpy as np  

from vision.utils.constants import *
from vision.utils.paths import *
from vision.utils.pcd_proccessing import *
from scipy.ndimage import maximum_filter

class FloorFinder:
    def __init__(self, pcd):
        self.pcd = pcd

    def dilate_inside_points_fast(self, grid):
        R = inside_dilation_radius
        
        has_topleft  = maximum_filter(grid, size=(R+1, R+1), origin=(-R//2, -R//2)) == 1
        has_topright = maximum_filter(grid, size=(R+1, R+1), origin=(-R//2, R//2)) == 1
        has_botleft  = maximum_filter(grid, size=(R+1, R+1), origin=(R//2, -R//2)) == 1
        has_botright = maximum_filter(grid, size=(R+1, R+1), origin=(R//2, R//2)) == 1
        
        mask = (has_topleft & has_topright & has_botleft & has_botright & (grid == 1)).astype(np.uint8)
        
        # This zeroes out the mask within R pixels of any border.
        mask[:R, :] = 0
        mask[-R:, :] = 0
        mask[:, :R] = 0
        mask[:, -R:] = 0
        
        FILL_SIZE = 9
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FILL_SIZE, FILL_SIZE))
        dilated_mask = cv2.dilate(mask, kernel)
        
        return np.bitwise_or(grid.astype(np.uint8), dilated_mask)

    def get_floor_grid(self):
        pcd = self.pcd

        #Ищем плоскость пола и точки в ней и вне неё
        plane_model, _ = pcd.segment_plane(distance_threshold = distance_threshold,
                                                       num_iterations = ransac_iterations,
                                                       ransac_n = 3)
        [a, b, c, d] = plane_model
        #строим нормаль к поверхности   
        normal = np.array([a, b, c])

        # Убираем слишком далекие от пола точки
        pcd_points = np.asarray(pcd.points)
        plane_abc = [a, b, c]
        close_mask = (np.abs(pcd_points[:, 1]) <= assertion_radus) & \
                     (np.abs(pcd_points @ plane_abc + d) <= max_dist)
                     
        close_points = pcd_points[close_mask]

        # Разделяем на inliers и outliers
        close_distances_ds = np.abs(np.dot(close_points, normal) + d)
        inlier_mask = close_distances_ds <= distance_threshold
        inliers = close_points[inlier_mask]
        outliers = close_points[~inlier_mask]

        #меняем направление нормали в сторону большего количества точек
        condition = np.dot(outliers, normal) + d < 0
        if len(outliers[condition]) < len(outliers) // 2:
            normal = -normal

        # берем 2 перпендикулярных вектора в плоскости
        v1 = [1, 0, 0]
        plane_v1 = project_to_plane([v1], plane_model)[0]
        plane_v2 = np.cross(plane_v1, normal)

        #проецируем точки пола на плоскость пола
        camera_coords =  [0, 0 ,0]
        inliers = np.vstack([inliers, camera_coords])
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
        camera_grid_coords = (x_indices[-1], y_indices[-1])
        cv2.circle(grid, 
                   center=camera_grid_coords,
                   radius=int(assertion_radus/grid_step),
                   color=1,
                   thickness=-1)

        grid = self.dilate_inside_points_fast(grid)

        # наращиваем точки пола
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_fill_size, morph_fill_size))
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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_obst_size, morph_obst_size))
        floor_obstacles = cv2.morphologyEx(floor_obstacles, cv2.MORPH_OPEN, kernel)

        #убираем маленькие компоненты препятствий
        components_info = cv2.connectedComponentsWithStats(
            (1 - floor_obstacles).astype(np.uint8), 
            connectivity=4, 
            ltype=cv2.CV_32S
        )
        [num_labels, labels, stats, centroids] = components_info
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                floor_obstacles[labels == i] = 1

        #убираем маленькие компоненты пола
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_fill_size, morph_fill_size))
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

        return floor_obstacles, camera_grid_coords