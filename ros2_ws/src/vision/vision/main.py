import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from vision.utils.floor_finder import FloorFinder
import cv2
import open3d as o3d
import time 
import threading
from vision.utils.paths import *

class DistanceServer(Node):
    def __init__(self):
        super().__init__('distance_server')
        self.floor_img = None
        self.latest_pcd = None
        self.gatekeeper_enabled = False

        srv_cb_group = MutuallyExclusiveCallbackGroup()
        sub_cb_group = MutuallyExclusiveCallbackGroup()

        self.srv = self.create_service(
            Trigger, 
            'get_distance', 
            self.get_distance_callback,
            callback_group=srv_cb_group
        )
        self.sub = self.create_subscription(
            PointCloud2, 
            '/zed/zed_node/point_cloud/cloud_registered', 
            self.pcd_callback, 
            1,
            callback_group=sub_cb_group
        )
        
        self.get_logger().info('Distance Service Server is ready.')

    def pcd_callback(self, msg):
        if not self.gatekeeper_enabled:
            return
        xyz = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        xyz = xyz[~np.isnan(xyz).any(axis=1)] 

        self.latest_pcd = xyz
        self.gatekeeper_enabled = False

    def get_distance_callback(self, request, response):
        self.latest_pcd = None
        timeout = 2.0
        
        self.gatekeeper_enabled = True
        start_time = time.time()
        while self.latest_pcd is None and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        self.gatekeeper_enabled = False

        if self.latest_pcd is None:
            response.success = False
            response.message = str(-1.0)
            self.get_logger().error('Timed out waiting for a point cloud frame.')
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.latest_pcd)
            
            finder = FloorFinder(pcd)
            floor, camera_grid_coords = finder.get_floor_grid()

            x_coord, y_coord = camera_grid_coords
            walking_path = floor[:, x_coord:]
            pixel_distance = np.argmax(np.any(walking_path == 0, axis = 0))
            distance = pixel_distance * grid_step

            self.floor_img = (floor * 255).astype(np.uint8)

            response.success = True
            response.message = str(distance)


        return response
    

def main():
    if not rclpy.ok():
        rclpy.init()
        
    distance_server = DistanceServer()
    
    executor = MultiThreadedExecutor()
    executor.add_node(distance_server)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        while rclpy.ok():
            if distance_server.floor_img is not None:
                cv2.imshow('floor', distance_server.floor_img)
            cv2.waitKey(50)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        distance_server.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()