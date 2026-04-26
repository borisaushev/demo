import math

from utils.constants import * 
import utils.reeds_shepp_path_planning as rs

import heapq
import math
import numpy as np

def find_path(start, end, grid):
    """
    A* Pathfinding optimized for 2D grids.
    
    Args:
        start: Tuple (x, y)
        end: Tuple (x, y)
        grid: 2D numpy array (0 = Obstacle, 1 = Walkable)
        
    Returns:
        List of tuples [(x, y), ...] representing the path.
        Returns empty list if no path found.
    """
    # Ensure integer coordinates
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    
    # Dimensions
    rows, cols = grid.shape
    
    # Priority Queue: Stores tuples of (f_score, g_score, x, y)
    # We include g_score to break ties (prefer paths explored further)
    open_set = []
    heapq.heappush(open_set, (0, 0, start[0], start[1]))
    
    # Data Structures
    came_from = {}  # To reconstruct path: key=(x,y), val=(parent_x, parent_y)
    g_score = {start: 0} # Cost from start to current node
    
    # 8-Way Movement (Diagonal allowed)
    # (dx, dy, cost)
    movements = [
        (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1),       # Cardinals
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414) # Diagonals
    ]

    while open_set:
        # Get node with lowest F score
        current_f, current_g, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)

        # Goal Check
        if current == end:
            return reconstruct_path(came_from, current)

        # Neighbor Search
        for dx, dy, cost in movements:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)

            # 1. Boundary Check
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            
            # 2. Obstacle Check (Note: grid[y][x])
            if grid[ny][nx] == 0:
                continue
            
            # 3. Calculate Scores
            tentative_g = g_score[current] + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # Found a better path to neighbor
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Heuristic: Euclidean Distance (Distance to Goal)
                # h = math.sqrt((end[0] - nx)**2 + (end[1] - ny)**2)
                # Optimization: Octile Distance is often preferred for 8-way grids, 
                # but Euclidean is sufficient and produces smoother-looking paths.
                h = math.hypot(end[0] - nx, end[1] - ny)
                
                f = tentative_g + h
                heapq.heappush(open_set, (f, tentative_g, nx, ny))
                
    return [] # No path found

def reconstruct_path(came_from, current):
    """Backtracks from end to start to build the path list."""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1] # Return reversed (Start -> End)


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
     