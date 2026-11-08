# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:50:38 2020
Update on 01/22/2021
@author: donghao Li
"""

import  os, time, logging,  shutil, glob
import roslaunch, rospy, rospkg, pathlib

import numpy as np
import pandas as pd
#from cpg_function import CPG
from gazebo_connection import GazeboConnection
# from gazebo_msgs.msg import ModelStates

class ROSLaunchParentFixed(roslaunch.parent.ROSLaunchParent):
    """
    Fixes Noetic bug: XmlRpcNode.shutdown() never calls server.shutdown(),
    leaving serve_forever() spinning at 100% CPU per shutdown cycle.
    Fix from: https://github.com/ros/ros_comm/issues/2238
    """
    def shutdown(self):
        server = self.server.server if self.server is not None and self.server.server is not None else None
        super(ROSLaunchParentFixed, self).shutdown()
        if server:
            server.shutdown()


class MubotEnv():

    def __init__(self,worker_folder,max_time, wk_id):
        print(f"\33[92mInitializing MubotEnv\033[0m")
        self._suppress_ros_console()
        rospack = rospkg.RosPack()
        self.worker_folder = worker_folder+"/"  #Database/worker_id
        self.max_sim_time = max_time
        self.wk_id = wk_id
        cli_args = ['des_mubot', 'main.launch', \
                    'urdf_robot_file:='+self.worker_folder+'mubot.urdf',\
                    'robot_name:=mubot_'+str(self.wk_id)]
        roslaunch_args = cli_args[2:]
        self.roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        mubot_launch = ROSLaunchParentFixed(uuid, self.roslaunch_file, is_core=False, verbose=False)
        ##############################################################
        # self._redirect_stdout()
        # try:
        #     mubot_launch.start()
        #     for i in range(4):
        #         print(i)
        #         time.sleep(1)
        #     self.gazebo = GazeboConnection()
        #     self.gazebo.unpauseSim()
        # finally:
        #     self._restore_stdout() 
        ##############
        mubot_launch.start()
        for i in range(4):
            print(i)
            time.sleep(1)
        self.gazebo = GazeboConnection()
        self.gazebo.unpauseSim()
        #####################################################33
        self._redirect_stdout()
        try:
            self.gazebo.pauseSim()
            mubot_launch.shutdown()
        finally:
            self._restore_stdout()
        #################
        # mubot_launch.shutdown()
        ##########################################################3
        print(f"\33[92mMubotEnv initialized\033[0m")
        self.close()
        return
        
    def close(self):
        self.gazebo.close()
        try:
            os.remove(self.worker_folder+'terminate.txt')
            os.remove(self.worker_folder+'drift_metrics.csv')
            os.remove(self.worker_folder+'control_signal.csv')
            os.remove(self.worker_folder+'velocity.csv')
        except FileNotFoundError:
            pass

        # self._force_kill_gazebo()
        return

    def step(self):
        #######################
        # Generate a fresh UUID each rollout — never reuse
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        #########################################
        self._redirect_stdout()   
        try:
            roslaunch.configure_logging(uuid)
            mubot_launch = ROSLaunchParentFixed(uuid, self.roslaunch_file, is_core=False, verbose=False)
            mubot_launch.start()
            time.sleep(4)
            self.gazebo.unpauseSim()
        finally:
            self._restore_stdout()
        ##################################
        # roslaunch.configure_logging(uuid)
        # mubot_launch = ROSLaunchParentFixed(uuid, self.roslaunch_file, is_core=False, verbose=False)
        # mubot_launch.start()
        # time.sleep(4)
        # self.gazebo.unpauseSim()
        ####################################333
        rospy.logdebug("Mubot launched")
        while not os.path.exists(self.worker_folder+'terminate.txt'):
            time.sleep(0.01)
        rospy.logdebug("Terminate file detected")
        self._redirect_stdout()
        try:
            self.gazebo.pauseSim()
            mubot_launch.shutdown()
        finally:
            self._restore_stdout()
        time.sleep(0.5)
        # reward = self.reward_calculation()
        # self.close()
        return #reward
    

    def gait_step(self):
        #######################
        # Generate a fresh UUID each rollout — never reuse
        cli_args = ['des_mubot', 'gait.launch', \
                    'urdf_robot_file:='+self.worker_folder+'mubot.urdf',\
                    'robot_name:=mubot_'+str(self.wk_id)]
        roslaunch_args = cli_args[2:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        mubot_launch = ROSLaunchParentFixed(uuid, roslaunch_file, is_core=False, verbose=False)
        mubot_launch.start()
        time.sleep(4)
        # self.gazebo.unpauseSim()
        rospy.logdebug("Mubot launched")
        # time.sleep(2)
        # rospy.init_node('mubot_evolve', anonymous=True, log_level=rospy.DEBUG)
        while not os.path.exists(self.worker_folder+'terminate.txt'):
            time.sleep(0.1)
        rospy.logdebug("Terminate file detected")
        self.gazebo.pauseSim()
        mubot_launch.shutdown()
        time.sleep(0.5)
        reward = self.reward_calculation()
        self.close()
        return reward

    
    def reward_calculation(self):
        df = pd.read_csv(self.worker_folder+"velocity.csv", header=None,names=["x_vel","y_vel","time","head_x"])
        df = df.dropna(subset=["time","x_vel","y_vel","head_x"])
        result = df[(df["time"] <= self.max_sim_time) & (df["time"] >= 4.0)]
        if not result.empty:
            reward = -result["x_vel"].mean()
            max_x = result["head_x"].max()
            if reward > 1.0:
                reward = -10.0
        else:
            reward = -1000.0
        return reward
    
    def reward(self):
        df = pd.read_csv(self.worker_folder+"drift_metrics.csv").to_numpy() # -spd, abs_lateral, rms_lat, heading_drift
        # print(df)
        self.close()
        return -df[0,0]-np.clip(df[0,1]-0.1,0.0,None)-np.clip(df[0,3]-0.1,0.0,None)*0.2


    def _suppress_ros_console(self):
        # Silence Python logging from roslaunch internals
        for name in ['roslaunch', 'roslaunch.parent', 'roslaunch.server',
                    'roslaunch.pmon', 'roslaunch.runner', 'roslaunch.child',
                    'rosmaster', 'rospy']:
            logging.getLogger(name).setLevel(logging.CRITICAL)

    def _redirect_stdout(self):
        """Redirect fd-level stdout/stderr to /dev/null.
        Catches output that bypasses Python logging (roslaunch C internals)."""
        self._devnull_fd = open(os.devnull, 'w')
        self._saved_stdout_fd = os.dup(1)   # save real stdout fd
        self._saved_stderr_fd = os.dup(2)   # save real stderr fd
        os.dup2(self._devnull_fd.fileno(), 1)  # redirect fd 1 → /dev/null
        os.dup2(self._devnull_fd.fileno(), 2)  # redirect fd 2 → /dev/null

    def _restore_stdout(self):
        """Restore stdout/stderr after roslaunch.start() completes."""
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        self._devnull_fd.close()

    def _cleanup_ros_logs(self):
        """Delete old ROS log directories, keeping only the most recent `keep` ones.
        Race-condition safe — another worker may delete dirs between glob and stat."""
        # log_dir = os.path.expanduser("~/.ros/log/")
        log_dir = f"/dev/shm/roslog_worker_{self.wk_id}/"
        # pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(log_dir):
            return

        def safe_getmtime(path):
            try:
                return os.path.getmtime(path)
            except FileNotFoundError:
                return 0.0  # already deleted by another worker — treat as oldest

        dirs = sorted(glob.glob(log_dir + "*/"), key=safe_getmtime)
        # print(dirs)
        for old_dir in dirs[:-1]:
            if old_dir[-7:] != 'latest/':
                try:
                    shutil.rmtree(old_dir)
                except FileNotFoundError:
                    pass  # another worker already cleaned it
                except Exception as e:
                    rospy.logwarn(f"[Worker {self.wk_id}] Could not remove log dir {old_dir}: {e}")

    
    # def _force_kill_gazebo(self):
    #     # """Hard-kill any gzserver/rosmaster left over from this worker's ports."""
    #     ros_port    = 10000 + self.wk_id
    #     gazebo_port = 20000 + self.wk_id

    #     # Kill gzserver/gzclient bound to this worker's Gazebo port
    #     subprocess.call( f"fuser -k {gazebo_port}/tcp 2>/dev/null || true", shell=True )
    #     # Kill rosmaster bound to this worker's ROS port
    #     subprocess.call( f"fuser -k {ros_port}/tcp 2>/dev/null || true", shell=True )
    #     # Belt-and-suspenders: kill any gzserver not responding on any port
    #     subprocess.call(
    #         "ps aux | grep gzserver | grep -v grep | "
    #         "awk '{print $2}' | xargs -r kill -9 2>/dev/null || true",
    #         shell=True
    #     )
    #     time.sleep(0.1)   # let OS reclaim ports before next launch

    # def _wait_for_gazebo_ready(self, timeout=10.0):
    #     robot_name = 'mubot_' + str(self.wk_id)
    #     deadline = time.time() + timeout

    #     # --- Stage 1: Gazebo services online ---
    #     srv = '/gazebo/unpause_physics'

    #     remaining = deadline - time.time()
    #     if remaining <= 0:
    #         raise RuntimeError(f"Gazebo service timeout waiting for {srv}")
    #     try:
    #         rospy.wait_for_service(srv)#, timeout=remaining)
    #     except rospy.ROSException:
    #         raise RuntimeError(f"Gazebo never became ready: {srv} not available")
    #     # print(f"Gazebo service {srv} is available")
    #     # --- Stage 2: Robot model spawned in world ---
    #     # # Poll /gazebo/model_states until our robot name appears.
    #     # rospy.logwarn(f"Waiting for {robot_name} to appear in Gazebo...")
    #     # while time.time() < deadline:
    #     #     try:
    #     #         msg = rospy.wait_for_message(
    #     #             '/gazebo/model_states', ModelStates, timeout=1.0
    #     #         )
    #     #         if robot_name in msg.name:
    #     #             rospy.logwarn(f"{robot_name} ready in {time.time()-( deadline - timeout):.2f}s")
    #     #             return  # ← Gazebo is fully ready
    #     #     except rospy.ROSException:
    #     #         pass  # topic not yet published, keep polling
    #     #     time.sleep(0.05)

    #     # raise RuntimeError(
    #     #     f"Timeout: {robot_name} never appeared in /gazebo/model_states "
    #     #     f"after {timeout}s — Gazebo may have failed to start"
    #     # )
    #     return