import cv2
import numpy as np  

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.heuristic import manhattan, euclidean

from utils.constants import * 

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


def pull_string(grid, path):
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
        
        add_line(smoothed_path, path[current_idx], path[best_visible_idx])
        current_idx = best_visible_idx
        
    return smoothed_path



def find_path(floor_walkable, start_pos, end_pos):
    grid = Grid(matrix=floor_walkable)
    
    start = grid.node(*start_pos)
    end = grid.node(*end_pos)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    
    path, runs = finder.find_path(start, end, grid)
    return path

