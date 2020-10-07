#!/usr/bin/env python3

import rospy
from openai_ros.robot_envs import turtlebot3_env
from gym import spaces
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from std_srvs.srv import Empty

import numpy as np

class TurtleBot3LocalizeEnv(turtlebot3_env.TurtleBot3Env):
    """
        TurtleBot3LocalizeEnv class is an implementation for localization of turtlebot3 task
    """

    def __init__(self):
        """
        Initialize TurtleBot3LocalizeEnv class

        Parameters
        ----------

        """
        super(TurtleBot3LocalizeEnv, self).__init__()

        rospy.loginfo('status: TurtleBot3LocalizeEnv is ready')

    def _check_amcl_data_is_ready(self):
        """
        Checks amcl is operational

        Returns
        -------

        """
        topic_name = '/particlecloud'
        topic_class = PoseArray
        time_out = 5.0
        self._particle_cloud = self._check_sensor_data_is_ready(topic_name, topic_class, time_out)

        topic_name = '/amcl_pose'
        topic_class = PoseWithCovarianceStamped
        time_out = 1.0
        self._amcl_pose = self._check_sensor_data_is_ready(topic_name, topic_class, time_out)

    def _init_amcl(self, is_global=True):
        """
        Initialize amcl

        Parameters
        ----------

        """

        if is_global:
            self._init_global_localization()

    def _init_global_localization(self):
        """
        Initialize global localization for amcl
        """

        service_name = '/global_localization'
        service_class = Empty
        self._call_service(service_name, service_class)

    def _set_init_pose(self):
        """
        Set the initial pose of the turtlebot3
        """

        pass

    def _init_env_variables(self):
        """
        Initialize environment variables
        """
        pass

    def _get_obs(self):
        """
        Return the observation from the environment
        """
        pass
