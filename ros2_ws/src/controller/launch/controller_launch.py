from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Define the vision node directly
    vision_node = Node(
        package='vision',              
        executable='distance_estimation',      
        name='distance_estimation',            
        output='screen'
    )

    # 2. Define your controller node
    controller_node = Node(
        package='controller',
        executable='controller',
        name='controller_node',
        output='screen'
    )

    # 3. Launch both nodes at the same time
    return LaunchDescription([
        vision_node,
        controller_node
    ])
