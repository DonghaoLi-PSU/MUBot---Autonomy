import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = 'mubot_sim'
    pkg_share = get_package_share_directory(pkg)
    pkg_prefix = get_package_prefix(pkg)

    real_launch = os.path.realpath(__file__)                # .../src/mubot_sim/launch/mubot_arg.launch.py
    src_pkg = os.path.dirname(os.path.dirname(real_launch)) # .../src/mubot_sim
    default_db = os.path.join(src_pkg, 'Database')
    if 'install' in src_pkg.split(os.sep):
        raise RuntimeError('Build with --symlink-install or pass database_dir:=...')

    database_dir = LaunchConfiguration('database_dir')
    robot_name = LaunchConfiguration('robot_name')

    set_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', os.path.join(pkg_prefix, 'lib', pkg))
    set_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', os.path.dirname(pkg_share))
    set_database = SetEnvironmentVariable('MUBOT_DATABASE_DIR', database_dir)

    args = [
        DeclareLaunchArgument(
            'database_dir', default_value=default_db,
            description='Root folder holding <model_id>/ subfolders'),
        DeclareLaunchArgument(
            'robot_name', default_value='mubot_1',
            description='Model name. Must start with "mubot_": plugins use name.substr(6) '
                        'as the Database sub-folder'),
        DeclareLaunchArgument(
            'urdf_path',
            default_value=[database_dir, '/',
                           PythonExpression(["'", robot_name, "'[6:]"]),
                           '/mubot.urdf'],
            description='Defaults to Database/<robot_name without "mubot_">/mubot.urdf'),
        DeclareLaunchArgument(
            'world', default_value=os.path.join(pkg_share, 'worlds', 'mubot.world'),
            description='World file (must NOT already <include> the robot)'),
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='true = server only (gz sim -s), no GUI'),
        DeclareLaunchArgument(
            'run', default_value='true',
            description="true = start unpaused (-r); false = start paused"),
    ]

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': [
                PythonExpression(["'-s ' if '", LaunchConfiguration('headless'),
                                  "' == 'true' else ''"]),
                PythonExpression(["'-r ' if '", LaunchConfiguration('run'),
                                  "' == 'true' else ''"]),
                '-v 3 ',
                LaunchConfiguration('world'),
            ],
            'on_exit_shutdown': 'true',
        }.items())

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-world', 'default',
            '-file', LaunchConfiguration('urdf_path'),
            '-name', robot_name,
            '-x', '0', '-y', '0', '-z', '0',
        ])

    # pause / unpause / step service for GazeboConnection
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=['/world/default/control@ros_gz_interfaces/srv/ControlWorld'])

    return LaunchDescription(
        [*args, set_plugin_path, set_resource_path, set_database, gz_sim, spawn, bridge])