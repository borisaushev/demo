# неблокирующая визуализация np массива
import cv2
import numpy as np  
import open3d as o3d
import open3d.visualization as vis
import time
import threading 
import tkinter as tk
from utils.paths import *

def visualise(np_arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_arr)
    show_pcd_concurrent(pcd)

# визуализация векторов
def visualise_vector_basis(pcd, normal, plane_v1, plane_v2):
    lines = []
    origin = (0,0,0)
    for vec, color in zip([normal, plane_v1, plane_v2], [[1,0,0], [0,1,0], [0,0,1]]):
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([origin, origin + np.array(vec)])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([color])
        lines.append(line)
    o3d.visualization.draw_geometries([pcd] + lines)


# просмотр pcd (вызывается в отдельном потоке функцией show_pcd_concurrent)
def show_pcd(pcd):
    window = vis.Visualizer()
    window.create_window(window_name="Open3d", width=1600, height=1200)
    window.add_geometry(pcd)
    while True:
        window.poll_events()
        window.update_renderer()
        time.sleep(0.001)
        try:
            if not window.poll_events():
                break
        except:
            break
    window.destroy_window()

# неблокирующий просмотр pcd
def show_pcd_concurrent(pcd):
    thread = threading.Thread(target=show_pcd, args=[pcd])
    thread.start()
    return thread

# просмотр изображений(np массивов)
def show_images(images_arr):
    root = tk.Tk()
    n = len(images_arr)
    w = root.winfo_screenwidth()
    root.destroy()
    gh, gw = images_arr[0].shape
    scale = (w/n) / gw
    new_w = int(gw * scale)
    new_h = int(gh * scale)

    cur_x = 0
    for i in range(n):
        win_name = 'Win' + str(i)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, new_w, new_h)
        cv2.moveWindow(win_name, cur_x, 0)
        cur_x += new_w
        resized_image = cv2.resize(images_arr[i], (new_w, new_h))
        cv2.imshow(win_name, (resized_image * 255).astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# неблокирующий просмотр изображений(np массивов)
def show_images_parallel(images_arr):
    thread = threading.Thread(target = show_images, args = [images_arr])
    thread.start()
    return thread

def mouse_callback(event, x, y, flags, param):
        grid = param[0]
        visualizer = param[1] 
        if event == cv2.EVENT_LBUTTONDOWN:
            if visualizer.count % 2 == 0:
                visualizer.start = (x, y)
            else:
                visualizer.end = (x, y)
            visualizer.count+=1
            grid_copy = np.copy(grid)
            cv2.circle(grid_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('floor_walkable', grid_copy)

class PathVisualizer:
    def __init__(self, grid, full_grid):
        self.start = None
        self.end = None
        self.count = 0
        self.grid = grid
        self.full_grid = full_grid

        cv2.imshow('floor_walkable', grid)
        cv2.setMouseCallback('floor_walkable', mouse_callback, (grid, self))
        cv2.waitKey(0)
        cv2.destroyAllWindows()   

    def find_path(self):
        path = find_path(self.grid, self.start, self.end)
        result_img = self.visualize_path(path)

        cv2.imshow('found path', result_img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()


    def visualize_path(self, path):
        vis_img = (self.full_grid * 255).astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        copy = (self.full_grid * 255).astype(np.uint8)
        copy = cv2.cvtColor(copy, cv2.COLOR_GRAY2BGR)
        
        if path:
            start_pos = (path[0].x, path[0].y)
            end_pos = (path[-1].x, path[-1].y)
            #TODO delete
            for point in path:
                cv2.circle(copy, (point.x, point.y), 2, (255, 0, 0), -1)

            cv2.circle(copy, start_pos, 5, (0, 255, 0), -1)
            cv2.circle(copy, end_pos, 5, (0, 0, 255), -1)

            cv2.imshow('Pathfinding', copy)
            cv2.waitKey(0)  
            cv2.destroyAllWindows()
            #TODO delete

            path = pull_string(self.grid, path)
            for point in path:
                cv2.circle(vis_img, (point.x, point.y), 2, (255, 0, 0), -1)

            cv2.circle(vis_img, start_pos, 5, (0, 255, 0), -1)
            cv2.circle(vis_img, end_pos, 5, (0, 0, 255), -1)
        else:
            print("No path found.")
            
        return vis_img

