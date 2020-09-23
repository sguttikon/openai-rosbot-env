#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty

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

        rospy.loginfo('status: gazebo connection establised')

    def reset_sim(self):
        """
        Using _reset_type variable's value the corresponding reset service
        of gazebo is called

        """

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
        self._call_service(service_name, service_class)


    def pause_sim(self):
        """
        Pause the physics updates of gazebo

        """

        service_name = '/gazebo/pause_physics'
        service_class = Empty
        self._call_service(service_name, service_class)


    def unpause_sim(self):
        """
        Resume the phsics update of gazebo

        """

        service_name = '/gazebo/unpause_physics'
        service_class = Empty
        self._call_service(service_name, service_class)


    def _call_service(self, service_name: str, service_class, max_retry: int = 10):
        """
        Create a service proxy for given service_name and service_class and
        call the service

        Parameters
        ----------
        service_name: str
            name of the service
        service_class:
            service type
        max_retry: int
            maximum number of times to retry calling the service

        """

        # wait until the service becomes available
        rospy.wait_for_service(service_name, timeout=None)
        # create callable proxy to the service
        service_proxy = rospy.ServiceProxy(service_name, service_class)

        is_call_successful = False
        counter = 0

        # loop until the counter reached max retry limit or
        # until the ros is shutdown or service call is successful
        while not is_call_successful and not rospy.is_shutdown():
            if counter < max_retry:
                try:
                    # call service
                    service_proxy()
                    is_call_successful = True
                except rospy.ServiceException as e:
                    # service call failed increment the counter
                    counter += 1
            else:
                # max retry count reached
                rospy.logerr('call to the service %s failed', service_name)
                break

if __name__ == '__main__':
    gazebo = GazeboConnection(reset_type = 'SIMULATION')
