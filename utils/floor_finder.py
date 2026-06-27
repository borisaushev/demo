import cv2
import numpy as np  
import open3d as o3d
import time

from utils.constants import *
from utils.paths import *
from utils.pcd_proccessing import *
from utils.visuals import *
from scipy.ndimage import maximum_filter

class FloorFinder:

    def dilate_inside_points_fast(self, grid):
        R = 8
        
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

    def find_floor_z_sliding_window(self, points):
        # Extract only Z coordinates and sort them
        z_coords = np.sort(points[:, 2])
        
        max_window_size = 2 * plane_distance_threshold
        left = 0
        best_left = 0
        best_right = 0
        max_count = 0
        
        # Slide the right pointer across the sorted Z coordinates
        for right in range(len(z_coords)):
            # Shrink window from the left if it exceeds the allowable width
            while z_coords[right] - z_coords[left] > max_window_size:
                left += 1
                
            # Track the window containing the most points
            current_count = right - left + 1
            if current_count > max_count:
                max_count = current_count
                best_left = left
                best_right = right
                
        # The optimal floor Z is the median of the densest window
        floor_z = np.median(z_coords[best_left:best_right + 1])
        return floor_z


    def get_floor_grid(self, points):
        floor_z = self.find_floor_z_sliding_window(points)

        z = points[:, 2]

        close_mask = (np.abs(z - floor_z) <= max_dist)
        close_points = points[close_mask]

        # Разделяем на inliers и outliers
        z = close_points[:, 2]
        inlier_mask = (z <= plane_distance_threshold)
        inliers = close_points[inlier_mask]
        outliers = close_points[~inlier_mask]

        #проецируем точки пола на плоскость пола
        plane_coords = inliers[:, :2]
        x_min, y_min = np.min(plane_coords, axis=0)
        x_max, y_max = np.max(plane_coords, axis=0)

        #строим двумерный массив, соответствующий плоскости
        width = int((x_max - x_min) // grid_step + 1)
        height = int((y_max - y_min) // grid_step + 1)
        grid = np.zeros((height, width), dtype=np.uint8)

        # Инвертируем Y т.к. в opencv 0 по оси Y это самый верх
        x_indices = ((plane_coords[:, 0] - x_min) // grid_step).astype(int)
        y_indices = ((y_max - plane_coords[:, 1]) // grid_step).astype(int) 
        valid_mask = (x_indices >= 0) & (x_indices < grid.shape[1]) & \
                    (y_indices >= 0) & (y_indices < grid.shape[0])

        grid[y_indices[valid_mask], x_indices[valid_mask]] = 1

        floor_points = np.copy(grid)
        grid = self.dilate_inside_points_fast(grid)

        # наращиваем точки пола
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_fill_size, morph_fill_size))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

        # убираем из плоскости точки, где есть препятствия
        outliers_plane_coords = outliers[:, :2]
        x_indices = ((outliers_plane_coords[:, 0] - x_min) // grid_step).astype(int)
        y_indices = ((y_max - outliers_plane_coords[:, 1]) // grid_step).astype(int)
        valid_mask = (x_indices >= 0) & (x_indices < grid.shape[1]) & \
                    (y_indices >= 0) & (y_indices < grid.shape[0])

        floor_obstacles = grid
        floor_obstacles[y_indices[valid_mask], x_indices[valid_mask]] = 0
        floor_points[y_indices[valid_mask], x_indices[valid_mask]] = 0        

        obstacles = np.copy(floor_obstacles)

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

        return floor_points, obstacles, floor_obstacles