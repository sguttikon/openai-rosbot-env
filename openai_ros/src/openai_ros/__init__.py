from gym.envs.registration import register

register(
        id='TurtleBot3World-v0',
        entry_point='openai_ros.task_envs.turtlebot3.turtlebot3_world:TurtleBot3WorldEnv'
)

register(
        id='TurtleBot3Localize-v0',
        entry_point='openai_ros.task_envs.turtlebot3.turtlebot3_localize:TurtleBot3LocalizeEnv'
)
