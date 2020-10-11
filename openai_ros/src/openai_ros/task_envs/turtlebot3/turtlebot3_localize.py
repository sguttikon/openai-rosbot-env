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
    """

    def __init__(self):
        """
        Initialize TurtleBot3LocalizeEnv class

        Parameters
        ----------

        """
        super(TurtleBot3LocalizeEnv, self).__init__()

        # code related to motion commands
        self._motion_error = 0.05
        self._update_rate = 30
        self._init_linear_speed = 0.0
        self._init_angular_speed = 0.0
        self._linear_forward_speed = 0.5
        self._linear_turn_speed = 0.05
        self._angular_speed = 1.0
        self._max_steps = 50
        self._dist_threshold = 0.1
        self._ent_threshold = -1.5
        self._front_threshold = 0.6
        self._left_threshold = self._right_threshold = 0.4

        self._robot = Robot()
        self._is_new_map = False
        self._robot_radius = 3.0
        self._episode_done = False
        self._current_step = 0
        # all sensor data, topic messages is assumed to be in same tf frame
        self._global_frame_id = 'map'
        self._scan_frame_id = 'base_scan'
        self._sector_angle = 30 # degrees view
        self._front_view = 120 # degrees view
        self._left_view = self._right_view = 60 # degrees view
        self._collision = np.zeros((360//self._sector_angle, 2), dtype=float) # anti-clockwise
        self._is_collision = False

        # code realted to sensors
        self._request_map = True
        self._request_laser = True
        self._request_odom = True
        self._request_imu = False
        self._request_amcl = True
        self._request_gazebo_data = True

        # code related to displaying results in matplotlib
        fig = plt.figure(figsize=(6, 6))
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
                self.__draw_surround_view(self._robot.get_pose(),
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
            particles = 20000
            client = dynamic_reconfig.Client('/amcl')
            config_params = {
                        'max_particles' : particles,
                     }
            config = client.update_configuration(config_params)

            self._init_global_localization()

        rospy.loginfo('status: amcl initialized')

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

    def _get_obs(self):
        """
        Return the observation from the environment
        """
        sqr_dist_err = self.__estimate_pose_error(self._robot.get_pose(), self._amcl_pose)
        self._amcl_pose.set_estimate_error(sqr_dist_err)

        return self._particle_cloud


    def _is_done(self):
        """
        Indicates whether or not the episode is done

        """

        # done if within distance threshold range with smallest entropy
        # or max steps elapsed
        if self._current_step > self._max_steps or \
            ( self._amcl_pose.get_estimate_error() < self._dist_threshold and \
                 ( np.isinf(self._amcl_pose.get_entropy()) or \
                     self._amcl_pose.get_entropy() < self._ent_threshold )
            ):
            self._episode_done = True

        return self._episode_done

    def _compute_reward(self, observation, done):
        """
        Calculate the reward based on the observation

        """

        if self._is_collision:
            reward = -10.0
        else:
            # to avoid division by zero
            #   sqr_error: error is always positive best value 0.0
            #   entropy: assuming 10e^-9 precision best value -5.0
            reward = 1/(self._amcl_pose.get_estimate_error() - self._dist_threshold + 1) + \
                    1/(self._amcl_pose.get_entropy() - self._ent_threshold + 5)

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

        # increment step counter
        self._current_step += 1
        self._is_collision = False
        linear_speed = 0.0
        angular_speed = 0.0
        if action == 0:     # move forward
            indices = [] #np.where(self._collision < self._forward_threshold)[0]
            if ( (0 not in indices) and (1 not in indices) and \
                 (10 not in indices) and (11 not in indices) ):
                linear_speed = self._linear_forward_speed
                angular_speed = 0.0
            else:
                rospy.logwarn('cannot execute action: 0, obstacle in path')
                self._is_collision = True
        elif action == 1:   # turn left
            indices = [] #np.where(self._collision < self._side_threshold)[0]
            if ( (2 not in indices) and (3 not in indices) ):
                linear_speed = self._linear_turn_speed
                angular_speed = self._angular_speed
            else:
                rospy.logwarn('cannot execute action: 1, obstacle in path')
                self._is_collision = True
        elif action == 2:   # turn right
            indices = [] #np.where(self._collision < self._side_threshold)[0]
            if ( (8 not in indices) and (9 not in indices) ):
                linear_speed = self._linear_turn_speed
                angular_speed = -1 * self._angular_speed
            else:
                rospy.logwarn('cannot execute action: 1, obstacle in path')
                self._is_collision = True
        else:               # do nothing / stop
            pass

        self._move_base( linear_speed, angular_speed,
                         self._motion_error, self._update_rate )

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

    def __draw_surround_view(self, robot_pose, surroundings_plt: dict):
        """
        Draw robot surroundings

        :param utils.Pose robot_pose: robot's pose
               dict surroundings_plt: plots of surroundings of robot
        :return dict
        """
        if robot_pose is None:
            return

        # get robot position
        scale = self._map_data.get_scale()
        pose_x, pose_y, _ = robot_pose.get_position()
        _, _, yaw = robot_pose.get_euler()

        # calculate sector angles -> anti-clockwise direction
        left_min = np.degrees(yaw) + self._front_view/2
        left_max = np.degrees(yaw) + self._front_view/2 + self._left_view
        left_color = 'silver'
        l_start = (self._front_view/2)//self._sector_angle
        l_end = (self._front_view/2 + self._left_view)//self._sector_angle

        right_min = np.degrees(yaw) - self._front_view/2 - self._right_view
        right_max = np.degrees(yaw) - self._front_view/2
        right_color = 'silver'
        r_start = (360 - self._front_view/2 - self._right_view)//self._sector_angle
        r_end = (360 - self._front_view/2)//self._sector_angle

        front_min = np.degrees(yaw) - self._front_view/2
        front_max = np.degrees(yaw) + self._front_view/2
        front_color = 'silver'
        f_start = (self._front_view/2)//self._sector_angle
        f_end = (360 - self._front_view/2)//self._sector_angle

        scan_beam = []
        for idx in range(len(self._collision)):
            beam = self._collision[idx]
            if idx>=l_start and idx<l_end:
                threshold = self._left_threshold
            elif idx>=r_start and idx<r_end:
                threshold = self._right_threshold
            elif idx<f_start or idx>=f_end:
                threshold = self._front_threshold
            else:
                threshold = -1

            if (not np.isinf(beam[0])) and beam[0] < threshold:
                x = [pose_x, pose_x + beam[0] * np.cos(beam[1]) / scale]
                y = [pose_y, pose_y + beam[0] * np.sin(beam[1]) / scale]
                if threshold == self._front_threshold:
                    front_color = 'red'
                elif threshold == self._right_threshold:
                    right_color = 'red'
                elif threshold == self._left_threshold:
                    left_color = 'red'
            else:
                x = [pose_x, pose_x]
                y = [pose_y, pose_y]
            scan_beam.append([x, y])
        scan_beam = np.asarray(scan_beam)

        if surroundings_plt['front'] == None:
            # left
            surroundings_plt['left'] = Wedge((pose_x, pose_y), self._left_threshold/scale,
                    left_min, left_max, color=left_color, alpha=0.5)
            self._plt_ax.add_artist(surroundings_plt['left'])
            # front
            surroundings_plt['front'] = Wedge((pose_x, pose_y), self._front_threshold/scale,
                    front_min, front_max, color=front_color, alpha=0.5)
            self._plt_ax.add_artist(surroundings_plt['front'])
            # right
            surroundings_plt['right'] = Wedge((pose_x, pose_y), self._right_threshold/scale,
                    right_min, right_max, color=right_color, alpha=0.5)
            self._plt_ax.add_artist(surroundings_plt['right'])

            for idx in range(len(self._collision)):
                beam_plot, = self._plt_ax.plot(scan_beam[idx, 0, :], scan_beam[idx, 1, :], color='k', alpha=0.5)
                surroundings_plt['sector_beams'].append(beam_plot)
        else:
            surroundings_plt['left'].update({'center': [pose_x, pose_y],
                                             'theta1': left_min,
                                             'theta2': left_max,
                                             'color': left_color})
            surroundings_plt['front'].update({'center': [pose_x, pose_y],
                                              'theta1': front_min,
                                              'theta2': front_max,
                                              'color': front_color})
            surroundings_plt['right'].update({'center': [pose_x, pose_y],
                                              'theta1': right_min,
                                              'theta2': right_max,
                                              'color': right_color})
            for idx in range(len(self._collision)):
                surroundings_plt['sector_beams'][idx].update({'xdata': scan_beam[idx, 0, :], 'ydata': scan_beam[idx, 1, :]})
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
                self._collision[:, ...].fill(np.inf)
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
                    if lrange < self._collision[collision_idx][0]:
                        self._collision[collision_idx][0] = lrange
                        self._collision[collision_idx][1] = langle
            else:
                # show scan w.r.t groundtruth pose
                self._check_gazebo_data_is_ready()  # get latest _gt_pose
                gt_x, gt_y, _ = self._robot.get_pose().get_position()
                _, _, gt_a = self._robot.get_pose().get_euler()
                scale = self._map_data.get_scale()

                collision_idx = 0
                self._collision[:, ...].fill(np.inf)
                for idx in range(len(scan_msg.ranges)):
                    lrange = scan_msg.ranges[idx]
                    if np.isinf(lrange):
                        continue
                    # if beam hits obstacle, store bean endpoint
                    langle = gt_a + scan_msg.angle_min + ( idx * scan_msg.angle_increment )
                    x = gt_x + lrange * np.cos(langle) / scale
                    y = gt_y + lrange * np.sin(langle) / scale
                    scan_points.append([x, y])

                    # store shortest beam range with angle per sector
                    collision_idx = idx//self._sector_angle
                    if lrange < self._collision[collision_idx][0]:
                        self._collision[collision_idx][0] = lrange
                        self._collision[collision_idx][1] = langle

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

    def __set_surroundings_details(self, robot):
        """
        Check the surroundings of robot

        :params utils.Robot robot: robot details
        :return utils.Robot
        """
        pass

    ###### private methods ######
