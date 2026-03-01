"""ROS2 Launch file per la demo fiera Agibot X2."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Percorso config
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'demo_config.yaml'
    )

    return LaunchDescription([
        # Argomenti configurabili
        DeclareLaunchArgument(
            'config_path',
            default_value=config_path,
            description='Percorso al file di configurazione YAML'
        ),

        LogInfo(msg="=" * 60),
        LogInfo(msg="  Lancio Demo Fiera Agibot X2 - Pick & Place Cialde"),
        LogInfo(msg="=" * 60),

        # Nodo principale demo
        Node(
            package='agibotx2_fiera',
            executable='demo_node',
            name='agibot_x2_demo_fiera',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
            }],
            # Rimappa topic se necessario
            remappings=[
                # ('/camera/color/image_raw', '/head_camera/color/image_raw'),
            ],
        ),
    ])
