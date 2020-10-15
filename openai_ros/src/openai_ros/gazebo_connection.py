#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose
import rospkg
import pojo
import utils
import time

class GazeboConnection():
    """
        GazeboConnection class handles all the interactions with the gazebo api
    """

    def __init__(self, reset_type: str):
        """
        Initialize GazeboConnection class

        Parameters
        ----------
        reset_type: str
            This paremeter is used within the reset_sim()
            Possible values are: ['SIMULATION', 'WORLD']

        """

        # reset the simulation
        self._reset_type = reset_type
        self.reset_sim()

        # HACK: pause the simulation
        self.pause_sim()

        # assuming by default we have ground_plane
        self.__init_models = ['ground_plane']
        self.__current_models = []
        self.__current_models.extend(self.__init_models)

        rospy.loginfo('status: gazebo connection establised')

    def reset_sim(self):
        """
        Using _reset_type variable's value the corresponding reset service
        of gazebo is called

        """

        rospy.logdebug('GazeboConnection.reset_sim() start')
        if self._reset_type == 'SIMULATION':
            # reset the entire simulation including the time
            service_name = '/gazebo/reset_simulation'
            service_class = Empty
        elif self._reset_type == 'WORLD':
            # reset the model's poses
            service_name = '/gazebo/reset_world'
            service_class = Empty
        else:
            # do nothing
            return
        utils.call_service(service_name, service_class)


    def pause_sim(self):
        """
        Pause the physics updates of gazebo

        """

        service_name = '/gazebo/pause_physics'
        service_class = Empty
        utils.call_service(service_name, service_class)


    def unpause_sim(self):
        """
        Resume the phsics update of gazebo

        """

        service_name = '/gazebo/unpause_physics'
        service_class = Empty
        utils.call_service(service_name, service_class)


    def spawn_sdf_model(self, model_name: str, initial_pose, robot_namespace: str = '', reference_frame: int = 'world'):
        """
        Spawns a model (*.sdf) to gazebo through service call

        :param str model_name: name of gazebo model to be spawn
               geometry_msgs.msg._Pose.Pose initial_pose: initial pose of model
               str robot_namespace: spawn model under this namespace
               str reference_frame: initial_pose is defined relative to the frame of this model,
                    default is gazebo world frame
        """

        if model_name in self.__current_models or \
            gazebo.get_model_state(model_name) is not None:
            rospy.logwarn('model: %s already exists in gazebo so will be respawned', model_name)
            self.delete_model(model_name)
            time.sleep(0.2)

        model_path = rospkg.RosPack().get_path('indoor_layouts') + '/models/'
        with open (model_path + model_name + '/model.sdf', 'r') as xml_file:
            # this should be an urdf or gazebo xml
            model_xml = xml_file.read().replace('\n', '')

        service_name = '/gazebo/spawn_sdf_model'
        service_class = SpawnModel
        service_req = SpawnModelRequest()
        service_req.model_name = model_name
        service_req.model_xml = model_xml
        service_req.initial_pose = pose
        service_req.robot_namespace = robot_namespace
        service_req.reference_frame = reference_frame

        response, is_successful = utils.call_service(service_name, service_class, service_req)
        if is_successful and response.success:
            # add model from tracking list
            self.__current_models.append(model_name)
            print(self.__current_models)
        else:
            rospy.logwarn(response.status_message)

    def delete_model(self, model_name: str):
        """
        Delete a model from gazebo through service call

        :param str model_name: name of gazebo model to be deleted
        """

        service_name = '/gazebo/delete_model'
        service_class = DeleteModel
        service_req = DeleteModelRequest()
        service_req.model_name = model_name

        response, is_successful = utils.call_service(service_name, service_class, service_req)
        if is_successful and response.success:
            # remove model from tracking list
            if model_name in self.__current_models:
                self.__current_models.remove(model_name)
        else:
            rospy.logwarn(response.status_message)

    def set_model_state(self, model_state):
        """
        Sets the model state in gazebo through service call

        :param gazebo_msgs.msg._ModelState.ModelState model_state: gazebo model pose and twist
        :return response received from service call
        """

        service_name = '/gazebo/set_model_state'
        service_class = SetModelState
        service_req = SetModelStateRequest()
        service_req.model_state = model_state

        response, is_successful = utils.call_service(service_name, service_class, service_req)
        if is_successful and response.success:
            pass # do nothing
        else:
            rospy.logwarn(response.status_message)

    def get_model_state(self, model_name: str, relative_entity_name: str = 'world'):
        """
        Gets the model state from gazebo through service call

        :param str model_name: name of gazebo model
               str relative_entity_name: return pose and twist relative to this entity,
                    default is gazebo world frame
        :return pojo.Pose
        """

        service_name = '/gazebo/get_model_state'
        service_class = GetModelState
        service_req = GetModelStateRequest()
        service_req.model_name = model_name
        service_req.relative_entity_name = relative_entity_name

        response, is_successful = utils.call_service(service_name, service_class, service_req)
        pose = None
        if is_successful and response.success:
            pose = pojo.Pose()
            pose._frame_id = response.header.frame_id
            pose.set_position(response.pose.position.x,
                              response.pose.position.y,
                              response.pose.position.z)
            pose.set_quaternion(response.pose.orientation.x,
                                response.pose.orientation.y,
                                response.pose.orientation.z,
                                response.pose.orientation.w)
            # TODO: twist ??

        else:
            rospy.logwarn(response.status_message)

        return pose

    def clear_all_spawned_models(self):
        """
        Clears all models that are not in __init_models list
        """

        topic_name = '/gazebo/model_states'
        topic_class = ModelStates
        data = utils.receive_topic_msg(topic_name, topic_class)

        if data is not None:
            for model_name in data.name:
                # if not a initial model delete
                if model_name not in self.__init_models:
                    self.delete_model(model_name)

if __name__ == '__main__':
    rospy.init_node('gazebo_connection')

    gazebo = GazeboConnection(reset_type = 'SIMULATION')
    # pose = Pose()
    # pose.position.x = 1
    # pose.position.y = 1
    # pose.position.z = 1
    # gazebo.spawn_sdf_model('sample',pose)
    #gazebo.delete_model('sample')

    # model_state = ModelState()
    # model_state.model_name = 'sample'
    # model_state.pose.orientation.w = 1
    # gazebo.set_model_state(model_state)
    #gazebo.get_model_state('sample')
