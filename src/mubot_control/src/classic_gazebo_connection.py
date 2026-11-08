# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass
#!/usr/bin/env python

import rospy, time
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from gazebo_msgs.srv import GetPhysicsProperties

from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class GazeboConnection():

    def __init__(self):
        
        # print(f"\33[92m Waiting for Gazebo services\033[0m")
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # print(f"\33[92m Unpause service is available\033[0m")

        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        # print(f"\33[92m Pause service is available\033[0m")

        rospy.wait_for_service('/gazebo/set_physics_properties')
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        # print(f"\33[92m Set physics properties service is available\033[0m")
        
        # self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        # self.reset_wo = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        # print(f"\33[92m Gazebo services are ready\033[0m")

        self.pauseSim()
        return

    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")

    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")

#     def resetSim(self):
#         rospy.wait_for_service('/gazebo/reset_simulation')
#         try:
#             self.reset_proxy()
# #            self.reset_wo
#         except rospy.ServiceException as e:
#             print ("/gazebo/reset_simulation service call failed")

#     def resetWorld(self):
#         rospy.wait_for_service('/gazebo/reset_world')
#         try:
#             self.reset_proxy()
#             self.reset_wo
#         except rospy.ServiceException as e:
#             print ("/gazebo/reset_world service call failed")

    def close(self):
        for proxy in  [self.unpause, self.pause, self.set_physics]:
            try:
                proxy.close()
            except Exception:
                pass

        time.sleep(0.1)
