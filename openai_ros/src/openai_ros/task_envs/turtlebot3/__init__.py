import sys

def set_path(path: str):
    try:
        sys.path.index(path)
    except ValueError:
        sys.path.insert(0, path)

# set programatically the path to 'openai_ros' directory (alternately can also set PYTHONPATH)
set_path('/media/suresh/research/awesome-robotics/active-slam/catkin_ws/src/openai-rosbot-env/openai_ros/src')
from openai_ros.task_envs.turtlebot3.turtlebot3_world import TurtleBot3WorldEnv
from openai_ros.task_envs.turtlebot3.turtlebot3_localize import TurtleBot3LocalizeEnv
