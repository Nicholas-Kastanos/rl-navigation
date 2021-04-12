from gym.envs.registration import register


register(
    id='TurtleBot3Navigation-v0',
    entry_point='gym_envs.task_envs.turtlebot3.navigation:TurtleBot3NavigationEnv',
    max_episode_steps=10000
)