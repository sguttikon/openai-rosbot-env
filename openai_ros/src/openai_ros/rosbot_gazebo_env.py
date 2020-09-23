#!/usr/bin/env python3

import rospy
import gym
from gym.utils import seeding
from openai_ros.gazebo_connection import GazeboConnection

class RosbotGazeboEnv(gym.Env):
    """
        RosbotGazeboEnv class acts as abstract gym environment template
    """

    def __init__(self, reset_type: str = 'SIMULATION'):
        """
        Initialize RosbotGazeboEnv class

        Parameters
        ----------
        reset_type: str
            This paremeter is used for creating GazeboConnection instance
            Possible values are: ['SIMULATION', 'WORLD']

        """

        # create GazeboConnection instance
        self.gazebo = GazeboConnection(reset_type = reset_type)

    def reset(self):
        """
        Override gym environment reset() with custom logic

        Returns
        -------
        obs:
            observation from the environment

        """

        # reset the gazebo simulation
        self._reset_sim()
        # set environment variables each time we reset
        self._init_env_variables()
        # get latest observation
        obs = self._get_obs()
        rospy.loginfo('status: environment is reset')
        return obs

    def close(self):
        """
        Override gym environment close() with custom logic

        """

        # initiate node shutdown
        rospy.signal_shutdown('closing RosbotGazeboEnv')

    def render(self):
        """
        Override gym environment render() with custom logic

        """

        super(RosbotGazeboEnv, self).render(mode='human')

    def step(self, action):
        """
        Override gym environment step() with custom logic

        Parameters
        ----------
        action:
            action to be executed in the environment

        Returns
        -------
        obs:
            observation from the environment
        reward:
            amount of reward achieved by taking the action
        done:
            indicate whether or not episode is done
        info:
            diagnostic information for debugging

        """

        # execute the action
        self.gazebo.unpause_sim()
        self._set_action(action)
        self.gazebo.pause_sim()

        # compute the required fields
        obs = self._get_obs()
        done = self._is_done()
        reward = self._compute_reward(obs, done)
        info = {}

        return obs, reward, done, info

    # ===== =====

    def _reset_sim(self):
        """
        Custom logic to reset the gazebo simulation

        """

        # pre-reset tasks
        self.gazebo.unpause_sim()
        self._check_all_systems_are_ready()
        self._set_init_pose()
        self.gazebo.pause_sim()

        # reset the gazebo
        self.gazebo.reset_sim()

        # check if everything working fine after reset
        self.gazebo.unpause_sim()
        self._check_all_systems_are_ready()
        self.gazebo.pause_sim()

    def _init_env_variables(self):
        """
        Initialize environment variables
        """
        raise NotImplementedError()

    def _set_init_pose(self):
        """
        Set the initial pose of the rosbot
        """
        raise NotImplementedError()

    def _get_obs(self):
        """
        Return the observation from the environment
        """
        raise NotImplementedError()

    def _check_all_systems_are_ready(self):
        """
        Checks all sensors and other simulation systems are operational
        """
        raise NotImplementedError()

    def _is_done(self):
        """
        Indicates whether or not the episode is done
        """
        raise NotImplementedError()

    def _compute_reward(self, observation, done):
        """
        Calculate the reward based on the observation
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """
        Apply the give action to the environment
        """
        raise NotImplementedError()

if __name__ == '__main__':
    env = RosbotGazeboEnv()
