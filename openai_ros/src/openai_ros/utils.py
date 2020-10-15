#!/usr/bin/env python3

import rospy

def call_service(service_name: str, service_class, service_req = None, time_out: float = 5, max_retry: int = 5):
    """
    Create a service proxy for given service_name and service_class and
    call the service

    :param str service_name: name of the service
           service_class: service type
           service_req: service request
           float time_out: timeout in seconds
           int max_retry: maximum number of times to retry calling the service
    :return response received from service call
            bool
    """

    # wait until the service becomes available
    try:
        rospy.wait_for_service(service_name, timeout = time_out)
    except rospy.ROSException as e:
        rospy.logerr('service %s is not available due to %s', service_name, e)
        return

    # create callable proxy to the service
    service_proxy = rospy.ServiceProxy(service_name, service_class)

    is_call_successful = False
    counter = 0
    response = None

    # loop until the counter reached max retry limit or
    # until the ros is shutdown or service call is successful
    while not is_call_successful and not rospy.is_shutdown():
        if counter < max_retry:
            try:
                # call service
                if service_req is None:
                    response = service_proxy()
                else:
                    response = service_proxy(service_req)
                is_call_successful = True
            except rospy.ServiceException as e:
                # service call failed increment the counter
                counter += 1
        else:
            # max retry count reached
            rospy.logerr('call to the service %s failed', service_name)
            break

    return response, is_call_successful

def receive_topic_msg(topic_name: str, topic_class, time_out: float = 5, max_retry: int = 5):
    """
    Check whether the topic is operational by
        1. subscribing to topic_name
        2. receive one topic_class message
        3. unsubscribe

    :param str topic_name: name of the topic
           topic_class: topic type
           float time_out: timeout in seconds
           int max_retry: maximum number of times to retry waiting for message
    :return rospy.Message
    """

    counter = 0
    response = None
    # loop until the ros is shutdown or received successfully message from topic
    while response is None and not rospy.is_shutdown():
        if counter < max_retry:
            try:
                # create a new subscription to topic, receive one message and then unsubscribe
                response = rospy.wait_for_message(topic_name, topic_class, timeout = time_out)
            except rospy.ROSException as e:
                counter += 1
        else:
            # max retry count reached
            rospy.logerr('wait for message from topic %s failed', topic_name)
            break

    return response
