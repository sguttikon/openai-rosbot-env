#!/usr/bin/env python3

import rospy
from openai_ros.robot_envs import turtlebot3_env
from openai_ros.utils import Map, Pose, Robot
from gym import spaces
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, PointStamped
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from nav_msgs.srv import GetMap
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import dynamic_reconfigure.client as dynamic_reconfig
import tf
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.patches import Ellipse
from matplotlib import transforms
import numpy as np

class TurtleBot3LocalizeEnv(turtlebot3_env.TurtleBot3Env):
    """
        TurtleBot3LocalizeEnv class is an implementation for localization of turtlebot3 task
        Goal is to become more certain about the position of turtlebot3
    """

    def __init__(self):
        """
        Initialize TurtleBot3LocalizeEnv class

        Parameters
        ----------

        """
        super(TurtleBot3LocalizeEnv, self).__init__(reset_type = 'WORLD')

        # TODO: need to get variable values from config file

        # for laser scan
        self._max_laser_value = 6
        self._min_laser_value = 0
        self._scan_ranges = []
        scan_high = np.full(360, self._max_laser_value)
        scan_low = np.full(360, self._min_laser_value)

        # for particle cloud [x_max, y_max, theta_max] for 384 x 384 map
        amcl_pose_high = np.array([384, 384, 1.0], dtype=np.float32)

        num_actions = 3
        self.action_space = spaces.Discrete(num_actions)
        self.reward_range = (-np.inf, np.inf)
        self.observation_space = spaces.Box(scan_low, scan_high)

        # code related to motion commands
        self._motion_error = 0.05
        self._update_rate = 30
        self._init_linear_speed = 0.0
        self._init_angular_speed = 0.0
        self._linear_forward_speed = 0.4
        self._linear_turn_speed = 0.05
        self._angular_speed = 0.75
        self._last_action = None
        self._forward_reward = 0.1
        self._turn_reward = 0.05

        self._dist_threshold = 10.0
        self._ent_threshold = -1.0

        self._robot = Robot()
        self._sector_angle = self._robot._sector_angle
        self._robot_radius = self._robot._robot_radius
        self._sector_laser_scan = np.zeros((360//self._sector_angle, 2), dtype=float) # anti-clockwise

        self._is_new_map = False
        self._episode_done = False
        self._current_step = 0
        self._max_steps = 100
        self._abort_episode = False
        self._success_episode = False
        self._cumulated_reward = []

        # all sensor data, topic messages is assumed to be in same tf frame
        self._global_frame_id = 'map'
        self._scan_frame_id = 'base_scan'
        self._collision_action = False

        # code realted to sensors
        self._request_map = True
        self._request_laser = True
        self._request_odom = True
        self._request_imu = False
        self._request_amcl = True
        self._request_gazebo_data = True

        # code related to displaying results in matplotlib
        fig = plt.figure(figsize=(7, 7))
        self._plt_ax = fig.add_subplot(111)
        plt.ion()
        plt.show()
        self._map_plt = None
        self._gt_pose_plts = {
            'robot': None,
            'heading': None,
            'surroundings': {
                'front': None,
                'left': None,
                'right': None,
                'back': None,
                'sector_beams': [],
            },
        }
        self._amcl_pose_plts = {
            'robot': None,
            'heading': None,
            'confidence': None,
        }
        self._scan_plt = None

        rospy.loginfo('status: TurtleBot3LocalizeEnv is ready')

    def render(self, mode='human'):
        """
        render the output in matplotlib plots
        """

        if self._map_data is not None:
            # environment map
            self.__draw_map(self._map_data)

            # groundtruth pose
            self._gt_pose_plts['robot'], self._gt_pose_plts['heading'] = \
                self.__draw_robot_pose(self._robot.get_pose(),
                                      self._gt_pose_plts['robot'],
                                      self._gt_pose_plts['heading'], 'blue')

            # groundtruth pose surroundings
            self._gt_pose_plts['surroundings'] = \
                self.__draw_surround_view(self._robot,
                                        self._gt_pose_plts['surroundings'])

            # amcl pose
            self._amcl_pose_plts['robot'], self._amcl_pose_plts['heading'] = \
                self.__draw_robot_pose(self._amcl_pose,
                                      self._amcl_pose_plts['robot'],
                                      self._amcl_pose_plts['heading'], 'green')
            # amcl pose covariance
            self._amcl_pose_plts['confidence'] = \
                self.__draw_pose_confidence(self._amcl_pose,
                                            self._amcl_pose_plts['confidence'], 'green')

            # laser scan
            self._scan_plt = \
                self.__draw_laser_scan(self._laser_scan, self._scan_plt, 'C0')

            self._plt_ax.legend([ self._gt_pose_plts['robot'], self._amcl_pose_plts['robot'], self._scan_plt ], \
                                [ 'gt_pose', 'amcl_pose', 'laser_scan' ])
        plt.draw()
        plt.pause(0.00000000001)

    def close(self):
        """
        Override turtlebot3 environment close() with custom logic
        """
        super(TurtleBot3LocalizeEnv, self).close()

        # prevent plot from closing after environment is closed
        plt.ioff()
        plt.show()

    def _check_amcl_data_is_ready(self):
        """
        Checks amcl topics are operational
        """

        topic_name = '/particlecloud'
        topic_class = PoseArray
        time_out = 5.0
        particle_msg = self._check_topic_data_is_ready(topic_name, topic_class, time_out)

        if particle_msg is not None:
            if particle_msg.header.frame_id != self._global_frame_id:
                rospy.logwarn('received amcl particle cloud must be in the global frame')

            self._particle_cloud = self.__process_particle_msg(particle_msg.poses)

        topic_name = '/amcl_pose'
        topic_class = PoseWithCovarianceStamped
        time_out = 1.0
        pose_msg = self._check_topic_data_is_ready(topic_name, topic_class, time_out)

        if pose_msg is not None:
            if pose_msg.header.frame_id != self._global_frame_id:
                rospy.logwarn('received amcl pose must be in the global frame')

            self._amcl_pose = self.__process_pose_cov_msg(pose_msg.pose)
            # rescale robot position
            x, y, z = self._amcl_pose.get_position() / self._map_data.get_scale()
            self._amcl_pose.set_position(x, y, z)

    def _check_gazebo_data_is_ready(self):
        """
        Checks gazebo topic is operational
        """

        topic_name = '/gazebo/model_states'
        topic_class = ModelStates
        time_out = 1.0
        data = self._check_topic_data_is_ready(topic_name, topic_class, time_out)

        # TODO: do we also need twist (velocity) of turtlebot ??
        # preprocess received data
        if data is not None:
            rosbot_name = 'turtlebot3'
            if rosbot_name in data.name:
                turtlebot_idx = data.name.index(rosbot_name)
                gt_pose = self.__process_pose_msg(data.pose[turtlebot_idx])
                self._robot.set_pose(gt_pose, self._map_data.get_scale())
            else:
                rospy.logwarn('cannot retrieve ground truth pose')

    def _laser_scan_callback(self, data):
        """
        Override turtlebot3 environment _laser_scan_callback() with custom logic
        """
        pass

    def _check_laser_scan_is_ready(self):
        """
        Override turtlebot3 environment _check_laser_scan_is_ready() with custom logic
        """

        topic_name = '/scan'
        topic_class = LaserScan
        time_out = 1.0
        data = self._check_topic_data_is_ready(topic_name, topic_class, time_out)

        if data is not None:
            self._laser_scan = self.__process_laser_msg(data)

    def _check_init_pose_pub_ready(self):
        """
        Checks initial pose publisher is operational
        """
        self._check_publisher_is_ready(self._init_pose_pub)

    def _check_map_data_is_ready(self):
        """
        Checks map service is operational
        """

        service_name = '/static_map'
        service_class = GetMap
        msg = self._call_service(service_name, service_class)

        if msg.map.header.frame_id != self._global_frame_id:
            rospy.logwarn('received map must be in the global frame')

        self._map_data = self.__process_map_msg(msg.map)

    def _init_amcl(self, is_global=True):
        """
        Initialize amcl

        Parameters
        ----------
        is_global: bool
            flag to initialize global localization or not
        """

        # publish initialpose for amcl
        init_pose_msg = PoseWithCovarianceStamped()
        init_pose_msg.header.stamp = rospy.get_rostime()
        init_pose_msg.header.frame_id = 'map'

        # position
        init_pose_msg.pose.pose.position.x = 0.0    # pose_x
        init_pose_msg.pose.pose.position.y = 0.0    # pose_y
        init_pose_msg.pose.pose.position.z = 0.0
        # orientation
        quaternion = quaternion_from_euler(0.0, 0.0, 0.0)   # pose_a
        init_pose_msg.pose.pose.orientation.x = quaternion[0]
        init_pose_msg.pose.pose.orientation.y = quaternion[1]
        init_pose_msg.pose.pose.orientation.z = quaternion[2]
        init_pose_msg.pose.pose.orientation.w = quaternion[3]
        # covariance
        covariance = [0.0]*36 # 6x6 covariance
        covariance[6*0 + 0] = 0.5 * 0.5 # cov_xx
        covariance[6*1 + 1] = 0.5 * 0.5 # cov_yy
        covariance[6*5 + 5] = (np.pi/12.0) *(np.pi/12.0)    # cov_aa
        init_pose_msg.pose.covariance = covariance

        self._init_pose_pub.publish(init_pose_msg)

        if is_global:
            # TODO: do we need selective resampling ??

            # dynamic reconfigure
            particles = 10000   # Note: only max 10000 is getting accepted
            client = dynamic_reconfig.Client('/amcl')
            config_params = {
                        'max_particles' : particles,
                     }
            config = client.update_configuration(config_params)

            self._init_global_localization()

        rospy.logdebug('status: amcl initialized')

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

        self._move_base( self._init_linear_speed, self._init_angular_speed,
                         self._motion_error, self._update_rate )

        self._init_amcl(is_global=True)

    def _init_env_variables(self):
        """
        Initialize environment variables
        """
        self._episode_done = False
        self._current_step = 0
        self._abort_episode = False
        self._success_episode = False
        self._cumulated_reward.clear()
        self._last_action = None

    def _get_obs(self):
        """
        Return the observation from the environment
        """
        sqr_dist_err = self.__estimate_pose_error(self._robot.get_pose(), self._amcl_pose)
        self._amcl_pose.set_estimate_error(sqr_dist_err)

        # update the surroundings of the robot
        self._robot.update_surroundings(self._sector_laser_scan)

        # return {
        #     'particle_cloud' : self._particle_cloud
        # }
        #return self._amcl_pose.get_position()
        return np.asarray(self._scan_ranges)


    def _is_done(self):
        """
        Indicates whether or not the episode is done

        """

        left_details = self._robot.get_surroundings()['left']
        back_details = self._robot.get_surroundings()['back']
        right_details = self._robot.get_surroundings()['right']
        front_details = self._robot.get_surroundings()['front']
        if self._robot.get_too_close() or (
            front_details['obstacle_sector'] == 1 and \
            back_details['obstacle_sector'] == 1 and \
                (left_details['obstacle_sector'] == 1 or \
                    right_details['obstacle_sector'] == 1)):
            # abort episode if robot is stuck or too close to obstacle
            self._abort_episode = self._episode_done = True
        elif self._current_step > self._max_steps or \
            ( self._amcl_pose.get_estimate_error() < self._dist_threshold and \
                 ( np.isinf(self._amcl_pose.get_entropy()) or \
                     self._amcl_pose.get_entropy() < self._ent_threshold )
            ):
            # episode done if within distance threshold range with smallest entropy
            # or max steps elapsed
            self._success_episode = self._episode_done = True
        else:
            self._episode_done = False

        return self._episode_done

    def _compute_reward(self, observation, done):
        """
        Calculate the reward based on the observation

        """

        if self._collision_action:
            reward = -0.01
        elif self._abort_episode:
            # penalty
            reward = -0.5
        elif self._success_episode:
            # bonus reward if successful
            reward = 5.0
        else:
            # to avoid division by zero
            #   sqr_error: error is always positive best value 0.0
            #   entropy: assuming 10e^-9 precision best value -5.0
            dist_reward = 1 / (self._amcl_pose.get_estimate_error() - self._dist_threshold + 1)
            entropy_reward = 1 / (self._amcl_pose.get_entropy() - self._ent_threshold + 5)
            reward = dist_reward + entropy_reward
            if self._last_action == 0:
                reward += self._forward_reward
            else:
                reward += self._turn_reward

            self._cumulated_reward.append([self._amcl_pose.get_estimate_error(),
                                           self._amcl_pose.get_entropy(),
                                           self._amcl_pose.get_position(),
                                           self._robot.get_pose().get_position()])
        if self._episode_done:
            rospy.logdebug("episode ended with {0} steps and {1}".format(self._current_step, np.asarray(self._cumulated_reward)))

        return reward

    def _set_action(self, action: int):
        """
        Apply the give action to the environment

        Parameters
        ----------
        action: int
            based on the action id number corresponding linear and angular speed for the rosbot is set

        Action List:
        * 0 = MoveFoward
        * 1 = TurnLeft
        * 2 = TurnRight

        """

        if self._robot.get_too_close():
            rospy.logwarn('too close obstacle episode will be terminated')
            return

        # increment step counter
        self._current_step += 1
        self._collision_action = False
        linear_speed = 0.0
        angular_speed = 0.0

        left_details = self._robot.get_surroundings()['left']
        back_details = self._robot.get_surroundings()['back']
        right_details = self._robot.get_surroundings()['right']
        front_details = self._robot.get_surroundings()['front']

        if action == 0:     # move forward
            if front_details['obstacle_sector'] == 0:
                linear_speed = self._linear_forward_speed
                angular_speed = 0.0
            else:
                # obstacle in front sector
                rospy.logdebug('action: 0 not executed, will hit obstacle')
                self._collision_action = True
        elif action == 1:   # turn left
            if left_details['obstacle_sector'] == 0 and back_details['obstacle_sector'] == 0:
                linear_speed = self._linear_turn_speed
                angular_speed = self._angular_speed
            else:
                # obstacle in left or back sector
                rospy.logdebug('action: 1 not executed, will hit obstacle')
                self._collision_action = True
        elif action == 2:   # turn right
            if right_details['obstacle_sector'] == 0 and back_details['obstacle_sector'] == 0:
                linear_speed = self._linear_turn_speed
                angular_speed = -1 * self._angular_speed
            else:
                # obstacle in right or back sector
                rospy.logdebug('action: 2 not executed, will hit obstacle')
                self._collision_action = True
        else:   # do nothing / stop
            pass

        self._move_base( linear_speed, angular_speed,
                         self._motion_error, self._update_rate )

        self._last_action = action
        rospy.logdebug('action: {0}'.format(action))

    ###### private methods ######

    def __draw_map(self, map):
        """
        Draw environment map

        :param utils.Map map: map of robot's environment
        """

        if self._is_new_map:
            width, height = map.get_size()
            scale = map.get_scale()
            orign_x, orign_y, _ = map.get_origin().get_position()

            # offset the map to display correctly w.r.t origin
            x_max = width/2 + orign_x/scale
            x_min = -width/2 + orign_x/scale
            y_max = height/2 + orign_y/scale
            y_min = -height/2 + orign_y/scale
            extent = [x_min, x_max, y_min, y_max]

            if self._map_plt == None:
                self._map_plt = self._plt_ax.imshow(map.get_cells(),
                            cmap=plt.cm.binary, origin='lower', extent=extent)
                self._plt_ax.plot(orign_x, orign_y, 'm+', markersize=14)
                self._plt_ax.grid()
                self._plt_ax.set_xlim([x_min, x_max])
                self._plt_ax.set_ylim([y_min, y_max])

                ticks_x = np.linspace(x_min, x_max)
                ticks_y = np.linspace(y_min, y_max)
                self._plt_ax.set_xticks(ticks_x, ' ')
                self._plt_ax.set_yticks(ticks_y, ' ')

                self._plt_ax.set_xlabel('x coords')
                self._plt_ax.set_ylabel('y coords')
            else:
                pass

            self._is_new_map = False

    def __draw_robot_pose(self, robot_pose, pose_plt: Wedge, heading_plt, color: str):
        """
        Draw robot pose

        :param utils.Pose robot_pose: robot's pose
               matplotlib.patches.Wedge pose_plt: plot of robot position
               matplotlib.lines.Line2D heading_plt: plot of robot heading
               str color: color used to render robot position and heading

        :return matplotlib.patches.Wedge, matplotlib.lines.Line2D
        """
        if robot_pose is None:
            return

        # get robot position
        pose_x, pose_y, _ = robot_pose.get_position()
        _, _, yaw = robot_pose.get_euler()

        line_len = 3.0
        xdata = [pose_x, pose_x + (self._robot_radius + line_len) * np.cos(yaw)]
        ydata = [pose_y, pose_y + (self._robot_radius + line_len) * np.sin(yaw)]

        if pose_plt == None:
            pose_plt = Wedge((pose_x, pose_y), self._robot_radius, 0, 360, color=color, alpha=0.5)
            self._plt_ax.add_artist(pose_plt)
            heading_plt, = self._plt_ax.plot(xdata, ydata, color=color, alpha=0.5)
        else:
            pose_plt.update({'center': [pose_x, pose_y]})
            heading_plt.update({'xdata': xdata, 'ydata': ydata})

        return pose_plt, heading_plt

    def __draw_surround_view(self, robot, surroundings_plt: dict):
        """
        Draw robot surroundings

        :param utils.Robot robot: robot details
               dict surroundings_plt: plots of surroundings of robot
        :return dict
        """

        # get robot position
        scale = self._map_data.get_scale()
        pose_x, pose_y, _ = robot.get_pose().get_position()
        _, _, yaw = robot.get_pose().get_euler()

        scan_beams = robot.get_scan_beams()
        left_details = robot.get_surroundings()['left']
        back_details = robot.get_surroundings()['back']
        right_details = robot.get_surroundings()['right']
        front_details = robot.get_surroundings()['front']
        if surroundings_plt['front'] == None:
            # left
            surroundings_plt['left'] = Wedge((pose_x, pose_y), left_details['threshold']/scale,
                    left_details['min_angle'], left_details['max_angle'], color=left_details['sector_color'], alpha=0.5)
            self._plt_ax.add_artist(surroundings_plt['left'])
            # back
            surroundings_plt['back'] = Wedge((pose_x, pose_y), back_details['threshold']/scale,
                    back_details['min_angle'], back_details['max_angle'], color=back_details['sector_color'], alpha=0.5)
            self._plt_ax.add_artist(surroundings_plt['back'])
            # right
            surroundings_plt['right'] = Wedge((pose_x, pose_y), right_details['threshold']/scale,
                    right_details['min_angle'], right_details['max_angle'], color=right_details['sector_color'], alpha=0.5)
            self._plt_ax.add_artist(surroundings_plt['right'])
            # front
            surroundings_plt['front'] = Wedge((pose_x, pose_y), front_details['threshold']/scale,
                    front_details['min_angle'], front_details['max_angle'], color=front_details['sector_color'], alpha=0.5)
            self._plt_ax.add_artist(surroundings_plt['front'])

            for idx in range(len(scan_beams)):
                beam_plot, = self._plt_ax.plot(scan_beams[idx, 0, :], scan_beams[idx, 1, :], color='cyan', alpha=0.25)
                surroundings_plt['sector_beams'].append(beam_plot)
        else:
            surroundings_plt['left'].update({
                                            'center': [pose_x, pose_y],
                                            'theta1': left_details['min_angle'],
                                            'theta2': left_details['max_angle'],
                                            'color': left_details['sector_color'],
                                            })
            surroundings_plt['back'].update({
                                            'center': [pose_x, pose_y],
                                            'theta1': back_details['min_angle'],
                                            'theta2': back_details['max_angle'],
                                            'color': back_details['sector_color']
                                            })
            surroundings_plt['right'].update({
                                            'center': [pose_x, pose_y],
                                            'theta1': right_details['min_angle'],
                                            'theta2': right_details['max_angle'],
                                            'color': right_details['sector_color']
                                            })
            surroundings_plt['front'].update({
                                            'center': [pose_x, pose_y],
                                            'theta1': front_details['min_angle'],
                                            'theta2': front_details['max_angle'],
                                            'color': front_details['sector_color']
                                            })
            for idx in range(len(scan_beams)):
                surroundings_plt['sector_beams'][idx].update({
                                            'xdata': scan_beams[idx, 0, :],
                                            'ydata': scan_beams[idx, 1, :]})
        return surroundings_plt

    def __draw_pose_confidence(self, robot_pose, confidence_plt, color: str, n_std=1.0):
        """
        Draw confidence ellipse around the robot pose

        :param utils.Pose robot_pose: robot's pose
               matplotlib.patches.Wedge confidence_plt: plot of robot position confidence
               str color: color used to render robot position confidence
               float n_std: number of std to determine ellipse's radius

        :return matplotlib.patches.Ellipse
        """

        pose_x, pose_y, _ = robot_pose.get_position()
        covariance = robot_pose.get_covariance()
        scale = self._map_data.get_scale()
        # reference  https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
        # cov_xy / np.sqrt(cov_xx * cov_yy)
        pearson = covariance[0, 1]/np.sqrt(covariance[0, 0] * covariance[1, 1])

        # compute eigenvalues and rescale
        ell_radius_x = np.sqrt(1 + pearson) / scale
        ell_radius_y = np.sqrt(1 - pearson) / scale

        # compute mean and std
        scale_x = np.sqrt(covariance[0, 0] / scale) * n_std
        scale_y = np.sqrt(covariance[1, 1] / scale) * n_std
        mean_x, mean_y = pose_x, pose_y

        transform = transforms.Affine2D().rotate_deg(45) \
                                         .scale(scale_x, scale_y) \
                                         .translate(mean_x, mean_y)
        if confidence_plt == None:
            confidence_plt = Ellipse((0, 0), width=ell_radius_x, height=ell_radius_y,
                                     facecolor='none', edgecolor=color)
            confidence_plt.set_transform(transform + self._plt_ax.transData)
            self._plt_ax.add_artist(confidence_plt)
        else:
            confidence_plt.width = ell_radius_x
            confidence_plt.height = ell_radius_y
            confidence_plt.set_transform(transform + self._plt_ax.transData)

        return confidence_plt

    def __draw_laser_scan(self, laser_scan, scan_plt, color: str):
        """
        Draw laser scan data in the environment

        :param numpy.ndarray laser_scan: laser scan data
               matplotlib.collections.PathCollection scan_plt: plot of laser scan
        """

        scale = self._map_data.get_scale()
        xdata = laser_scan[:, 0]
        ydata = laser_scan[:, 1]
        if scan_plt == None:
            scan_plt = plt.scatter(xdata, ydata, s=14, c=color)
        else:
            scan_plt.set_offsets(laser_scan)

        return scan_plt

    def __process_particle_msg(self, particle_msg):
        """
        Process the particle cloud message

        :param list[geometry_msgs.msg._Pose.Pose] particle_msg: list of poses
        :return numpy.ndarray
        """

        poses = []
        for pose_msg in particle_msg:
            pose = self.__process_pose_msg(pose_msg)
            x, y, _ = pose.get_position()
            _, _, yaw = pose.get_euler()
            poses.append([x, y, yaw])

        poses = np.asarray(poses)
        return poses

    def __process_pose_cov_msg(self, pose_cov_msg):
        """
        Process the received pose message

        :param geometry_msgs.msg._PoseWithCovariance.PoseWithCovariance pose_cov_msg: pose with covariance message
        :return utils.Pose
        """

        # initialize pose
        pose = self.__process_pose_msg(pose_cov_msg.pose)
        # initialize covariance
        pose.set_covariance(
            pose_cov_msg.covariance
        )

        return pose

    def __process_pose_msg(self, pose_msg):
        """
        Process the received pose message

        :param geometry_msgs.msg._Pose.Pose pose_msg: pose message
        :return utils.Pose
        """

        # initialize pose
        pose = Pose()
        pose.set_position(
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z
        )
        pose.set_quaternion(
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        )
        return pose

    def __process_map_msg(self, msg_map):
        """
        Process the received map message

        :param nav_msgs.msg._OccupancyGrid.OccupancyGrid msg_map: map message
        :return utils.Map
        """

        # initialize map
        map = Map()
        map.set_scale(msg_map.info.resolution)
        map.set_size(msg_map.info.width, msg_map.info.height)
        map.set_origin(self.__process_pose_msg(msg_map.info.origin))

        # rescale and shift the map origin to world coordinates
        origin = map.get_origin()
        x, y, z = origin.get_position()
        width, height = map.get_size()
        scale = map.get_scale()
        origin.set_position(
                x + (width/2) * scale,
                y + (height/2) * scale,
                z
        )

        # set grid cells
        map.set_cells(msg_map.data)

        self._is_new_map = True
        self._request_map = False

        return map

    def __process_laser_msg(self, scan_msg):
        """
        Process the received laser scane message

        :param sensor_msgs.msg._LaserScan.LaserScan scan_msg: laser scan message
        :return numpy.ndarray
        """

        transform_scan = False
        scan_points = []
        if scan_msg.header.frame_id == self._scan_frame_id:
            # transform from _scan_frame_id to _global_frame_id
            if transform_scan:
                # show scan w.r.t amcl pose

                # check whether transform is available
                tf_listener = tf.TransformListener()
                now = rospy.Time(0)
                try:
                    tf_listener.waitForTransform(self._scan_frame_id,
                                                 self._global_frame_id,
                                                 now,
                                                 rospy.Duration(1.0))
                except Exception as e:
                    rospy.logwarn('cannot transform from {0} to {1}'.format(self._scan_frame_id, self._global_frame_id))
                    return []

                # transform available laser scan point to map frame point
                collision_idx = 0
                self._sector_laser_scan[:, ...].fill(np.inf)
                for idx in range(len(scan_msg.ranges)):
                    lrange = scan_msg.ranges[idx]
                    if np.isinf(lrange):
                        continue
                    langle = scan_msg.angle_min + ( idx * scan_msg.angle_increment )
                    scan_point = PointStamped()
                    scan_point.header.frame_id = self._scan_frame_id
                    scan_point.header.stamp = now
                    scan_point.point.x = lrange * np.cos(langle)
                    scan_point.point.y = lrange * np.sin(langle)
                    scan_point.point.z = 0.0

                    map_point = tf_listener.transformPoint(self._global_frame_id, scan_point)
                    x = map_point.point.x
                    y = map_point.point.y

                    if np.isnan(x) or np.isinf(x) or np.isnan(y) or np.isinf(y):
                        continue
                    else:
                        scale = self._map_data.get_scale()
                        scan_points.append([x/scale, y/scale])

                    # store shortest beam range with angle per sector
                    collision_idx = idx//self._sector_angle
                    if lrange < self._sector_laser_scan[collision_idx][0]:
                        self._sector_laser_scan[collision_idx][0] = lrange
                        self._sector_laser_scan[collision_idx][1] = langle
            else:
                # show scan w.r.t groundtruth pose
                self._check_gazebo_data_is_ready()  # get latest _gt_pose
                gt_x, gt_y, _ = self._robot.get_pose().get_position()
                _, _, gt_a = self._robot.get_pose().get_euler()
                scale = self._map_data.get_scale()

                collision_idx = 0
                self._scan_ranges.clear()
                self._sector_laser_scan[:, ...].fill(np.inf)
                for idx in range(len(scan_msg.ranges)):
                    lrange = scan_msg.ranges[idx]
                    if np.isinf(lrange):
                        self._scan_ranges.append(self._max_laser_value)
                        continue
                    elif np.isnan(lrange):
                        self._scan_ranges.append(self._min_laser_value)
                        continue
                    else:
                        self._scan_ranges.append(lrange)

                    # if beam hits obstacle, store bean endpoint
                    langle = gt_a + scan_msg.angle_min + ( idx * scan_msg.angle_increment )
                    x = gt_x + lrange * np.cos(langle) / scale
                    y = gt_y + lrange * np.sin(langle) / scale
                    scan_points.append([x, y])

                    # store shortest beam range with angle per sector
                    collision_idx = idx//self._sector_angle
                    if lrange < self._sector_laser_scan[collision_idx][0]:
                        self._sector_laser_scan[collision_idx][0] = lrange
                        self._sector_laser_scan[collision_idx][1] = langle

        scan_points = np.asarray(scan_points)
        return scan_points

    def __estimate_pose_error(self, pose1, pose2):
        """
        Calculate the squared euclidean distance between two pose + covariance

        :param utils.Pose pose1: pose1
               utils.Pose pose2: pose2
        :return float
        """

        # calculate squared euclidean in pose+covariance
        sqr_dist_err = np.linalg.norm( pose1.get_position() - pose2.get_position() )**2 + \
               np.linalg.norm( pose1.get_euler() - pose2.get_euler() )**2 + \
               np.linalg.norm( pose1.get_covariance() - pose2.get_covariance() )**2

        return sqr_dist_err

    ###### private methods ######
