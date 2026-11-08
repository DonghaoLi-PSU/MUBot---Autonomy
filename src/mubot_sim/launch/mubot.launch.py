import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration



def generate_launch_description():
    pkg = 'mubot_sim'
    pkg_share = get_package_share_directory(pkg)
    pkg_prefix = get_package_prefix(pkg)
    world = os.path.join(pkg_share, 'worlds', 'mubot.world')

    # set up file path for plugin
    set_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', os.path.join(pkg_prefix))
    set_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.dirname(pkg_share))

    # Default: GUI + run immediately + info-level messages (gzmsg).
    # Headless:  ros2 launch mubot_sim mubot.launch.py gz_args:="-s -r -v 3 <world>"
    gz_args = DeclareLaunchArgument(
        'gz_args', default_value=f'-r -v 2 {world}',
        description='Arguments passed to gz sim')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': LaunchConfiguration('gz_args'),
            'on_exit_shutdown': 'true',
        }.items())

    return LaunchDescription([set_plugin_path, set_resource_path, gz_args, gz_sim])
