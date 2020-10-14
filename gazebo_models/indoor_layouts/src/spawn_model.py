import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose


rospy.init_node('insert_object',log_level=rospy.INFO)

initial_pose = Pose()
initial_pose.position.x = 1
initial_pose.position.y = 1
initial_pose.position.z = 1

f = open('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/openai-rosbot-env/gazebo_models/indoor_layouts/models/sample/model.sdf','r')
sdff = f.read()

print('waiting for service')
rospy.wait_for_service('gazebo/spawn_sdf_model')
print('service available')
spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
spawn_model_prox("some_robo_name", sdff, "robotos_name_space", initial_pose, "world")
