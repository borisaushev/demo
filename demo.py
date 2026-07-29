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

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R


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
            self.latest_frame = None
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
            floor, camera_grid_coords = finder.get_floor_grid()

            radius_in_pixels = int(diameter / grid_step) // 2
            x_coord, y_coord = camera_grid_coords
            walking_path = floor[y_coord - radius_in_pixels : y_coord + radius_in_pixels, x_coord:]
            pixel_distance = np.argmax(np.any(walking_path == 0, axis = 0))
            distance = pixel_distance * grid_step

            # 2. Convert grid to 8-bit image array values (0 to 255)
            floor_img = (floor * 255).astype(np.uint8)

            # Calculate rolling time difference and compute current FPS
            current_time = time.perf_counter()
            time_diff = current_time - prev_time
            prev_time = current_time
            
            if time_diff > 0:
                fps = 1.0 / time_diff

            # Convert to color image so text can be rendered cleanly in color overlay
            display_img = cv2.cvtColor(floor_img, cv2.COLOR_GRAY2BGR)

            # Draw the FPS counter overlay onto the color canvas image matrix
            img_text = f"FPS: {fps:.1f} Dist: {distance:.1f}"
            cv2.putText(
                img=       display_img, 
                text=      img_text, 
                org=       (15, 35),
                fontFace=  cv2.FONT_HERSHEY_SIMPLEX,   
                fontScale= 1.0,                        
                color=     (0, 255, 0),                
                thickness= 2,                      
                lineType=  cv2.LINE_AA              
            )

            cv2.circle(
                img=       display_img,
                center=    camera_grid_coords,
                radius=    10,
                color=     (0, 255, 0),
                thickness= -1
            )
            
            # 4. Show the locked-size display canvas with text overlay
            cv2.imshow('floor', display_img)

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
