import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor # FIX: Explicitly handle the execution loop
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import threading
from utils.floor_finder import FloorFinder
import cv2
import open3d as o3d
import time 
from utils.paths import *
from utils.visuals import * 


class RealTimePointCloudStreamer(Node):
    def __init__(self):
        super().__init__('pc_streamer')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
        # Use a single atomic tracking variable instead of a growing list queue
        self.latest_frame = None
        self.lock = threading.Lock() # Prevents thread collisions between network and app loop
        
        # Use a Queue Size of 1 to tell ROS to drop old network frames instantly
        self.sub = self.create_subscription(
            PointCloud2, 
            '/zed/zed_node/point_cloud/cloud_registered', 
            self.cb, 
            1
        )
        
    def cb(self, msg):
        xyz = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        xyz = xyz[~np.isnan(xyz).any(axis=1)] 
        
        # Thread-safely overwrite with the newest data frame
        with self.lock:
            self.latest_frame = xyz

    def get_current_frame(self):
        # Safely pull and consume the absolute newest frame available right now
        with self.lock:
            frame = self.latest_frame
            self.latest_frame = None # Consume the frame so we don't process duplicates
            return frame


def main():
    if not rclpy.ok():
        rclpy.init()
        
    streamer = RealTimePointCloudStreamer()
    
    # FIX 1: Create a dedicated local executor instance instead of using the global shared one
    executor = SingleThreadedExecutor()
    executor.add_node(streamer)
    
    # FIX 2: Target executor.spin instead of rclpy.spin
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    
    try:
        # Define your target window size (e.g., 640x640 or 800x600 pixels)
        TARGET_WIDTH = 640
        TARGET_HEIGHT = 640

        # Initialize FPS calculation trackers
        prev_time = time.perf_counter()
        fps = 0.0

        # Infinite real-time execution loop
        while rclpy.ok():
            points = streamer.get_current_frame()
            
            if points is None:
                time.sleep(0.01) 
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            finder = FloorFinder(pcd)
            (points, obstacles, floor) = finder.get_floor_grid()
            
            corrected_floor = floor[::-1, ::-1]

            # 2. Convert grid to 8-bit image array values (0 to 255)
            floor_img = (corrected_floor * 255).astype(np.uint8)
            
            diameter_in_pixels = int(diameter / grid_step)
            corrected_floor = cv2.erode(corrected_floor, np.ones((diameter_in_pixels, diameter_in_pixels)))
            if not np.any(corrected_floor == 1):
                continue

            indeces = np.argwhere(corrected_floor > 0)
            y_axis = indeces[:, 0]
            x_axis = indeces[:, 1]
            y_max = np.max(y_axis)
            x_max = np.max(x_axis[y_axis == y_max])
            y_min = np.min(y_axis)
            x_min = np.min(x_axis[y_axis == y_min])
            
            path = find_path((x_max, y_max), (x_min, y_min), corrected_floor)
            if path is not None:
                path_img = visualize_path(corrected_floor, path)
            else: 
                path_img = visualize_path(corrected_floor, path = [])
            
            # 3. Rescale the matrix to a forced static window footprint
            # Note: OpenCV expects sizing order format as (Width, Height)
            rescaled_floor = cv2.resize(
                floor_img, 
                (TARGET_WIDTH, TARGET_HEIGHT), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Calculate rolling time difference and compute current FPS
            current_time = time.perf_counter()
            time_diff = current_time - prev_time
            prev_time = current_time
            
            if time_diff > 0:
                fps = 1.0 / time_diff

            # Convert to color image so text can be rendered cleanly in color overlay
            display_img = cv2.cvtColor(rescaled_floor, cv2.COLOR_GRAY2BGR)

            # Draw the FPS counter overlay onto the color canvas image matrix
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                display_img, 
                fps_text, 
                (15, 35),                   # Coordinate location (X, Y)
                cv2.FONT_HERSHEY_SIMPLEX,   # Font styling
                1.0,                        # Font size scale
                (0, 255, 0),                # Text color (B, G, R) = Green
                2,                          # Text outline thickness
                cv2.LINE_AA                 # Anti-aliasing quality flag
            )
            
            # 4. Show the locked-size display canvas with text overlay
            cv2.imshow('floor', display_img)
            cv2.imshow('path', path_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break

    except KeyboardInterrupt:
        print("Cell stopped by user.")
    finally:
        # FIX 3: Fully dismantle the background executor system first
        executor.shutdown()  # This breaks the executor.spin loop and terminates the background thread
        streamer.destroy_node()
        cv2.destroyAllWindows()

main()
