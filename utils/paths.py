import math

from utils.constants import * 
import utils.reeds_shepp_path_planning as rs

import heapq
import math
import numpy as np

def euclidean_distance(p1, p2):
    """Calculates the direct straight-line distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def greedy_best_first_search_euclidean(grid, start, goal):
    y_len = len(grid)
    x_len = len(grid[0])
    
    # Priority Queue elements format: (heuristic_cost, current_node)
    frontier = []
    heapq.heappush(frontier, (euclidean_distance(start, goal), start))
    
    came_from = {start: None}
    visited = {start}
    
    # Movement configurations: (dr, dc, step_cost)
    movements = [
        (0, 1), (0, -1), (1, 0),  (-1, 0), # Cardinals
        (1, 1), (1, -1), (-1, 1), (-1, -1) # Diagonals
    ]
    
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if current == goal:
            break
            
        x, y = current
        for dx, dy in movements:
            new_x = x + dx
            new_y = y + dy
            neighbor = (new_x, new_y)
            
            # Boundary control
            if 0 <= new_x < x_len and 0 <= new_y < y_len:
                # Obstacle verification and visit check
                if grid[new_y][new_x] != 0 and neighbor not in visited:
                    visited.add(neighbor)
                    h_cost = euclidean_distance(neighbor, goal)
                    heapq.heappush(frontier, (h_cost, neighbor))
                    came_from[neighbor] = current
                    
    return reconstruct_path(came_from, start, goal)

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return None
        
    current = goal
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def has_line_of_sight(grid, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    # Direction of movement
    step_x = 1 if x2 > x1 else -1
    step_y = 1 if y2 > y1 else -1
    
    # Error term for Bresenham's algorithm
    error = dx - dy
    
    x, y = x1, y1
    
    # Trace the line from p1 to p2
    while not (x == x2 and y == y2):
        # Check if current cell is an obstacle (excluding start and end)
        if (x != x1 or y != y1) and (x != x2 or y != y2):
            if grid[y][x] == 0:
                return False
        
        # Bresenham's algorithm step
        error2 = error * 2
        
        if error2 > -dy:
            error -= dy
            x += step_x
        
        if error2 < dx:
            error += dx
            y += step_y
    
    return True

def add_line(path, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    step_x = 1 if x2 > x1 else -1
    step_y = 1 if y2 > y1 else -1
    
    error = dx - dy
    x, y = x1, y1
    
    while not (x == x2 and y == y2):
        error2 = error * 2
        
        if error2 > -dy:
            error -= dy
            x += step_x
        
        if error2 < dx:
            error += dx
            y += step_y
        
        path.append((x, y))


def pull_string(grid, path, add_lines = False):
    if len(path) <= 2:
        return path

    x0, y0 = path[0]
    smoothed_path = [(x0, y0)]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # Check furthest possible point we can see from the current node
        best_visible_idx = current_idx + 1
        for look_ahead in range(len(path) - 1, current_idx, -1):
            if has_line_of_sight(grid, path[current_idx], path[look_ahead]):
                best_visible_idx = look_ahead
                break
        if add_lines:
            add_line(smoothed_path, path[current_idx], path[best_visible_idx])
        else:
            x, y = path[best_visible_idx]
            smoothed_path.append((x, y))
        current_idx = best_visible_idx
        
    return smoothed_path


def smooth_trajectory(path, max_curvature=grid_step/(diameter/2), step_size=0.2):
    """
    1. Converts 2D path points to 3D directed poses.
    2. Generates Reeds-Shepp curves between poses.
    3. Validates against the Grid Object (0=Obstacle).
    """
    
    # --- Step A: Convert Points (x,y) to Poses (x,y,yaw) ---
    poses = []
    for i in range(len(path) - 1):
        curr_node = path[i]
        next_node = path[i+1]
        
        # Calculate heading (yaw) pointing toward the next node
        # atan2(dy, dx) gives the angle in radians
        yaw = math.atan2(next_node[1] - curr_node[1], next_node[0] - curr_node[0])
        poses.append((curr_node[0], curr_node[1], yaw))
    
    # Handle the final goal (retain the last known orientation)
    poses.append((path[-1][0], path[-1][1], poses[-1][2]))

    # --- Step B: Generate Curves & Check Collisions ---
    final_x, final_y = [], []
    
    for i in range(len(poses) - 1):
        start = poses[i]
        end = poses[i+1]
        
        # CALL THE LIBRARY: Generate the raw curve coordinates
        # rs.reeds_shepp_path_planning(sx, sy, syaw, gx, gy, gyaw, maxc, step_size)
        px, py, _, _, _ = rs.reeds_shepp_path_planning(
            start[0], start[1], start[2],
            end[0], end[1], end[2],
            max_curvature, step_size
        )
        
        final_x.extend(px)
        final_y.extend(py)
    return list(zip(final_x, final_y))
     

def find_path(start, end, grid):
        if not np.any(grid == 1):
            return [] 
        path = greedy_best_first_search_euclidean(grid, start, end)
        if path is None:
            return None
        
        # reversing path to pull in the opposite direction
        path = pull_string(grid, path[::-1], add_lines = True )
        # reversing it back and pulling in the original direction 
        path = pull_string(grid, path[::-1], add_lines = False)

        return  smooth_trajectory(path)

