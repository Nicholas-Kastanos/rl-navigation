#!/usr/bin/env python
import time
from functools import reduce

import gym
import gym_envs
import matplotlib.pyplot as plt
import numpy as np
import rospkg
# ROS packages required
import rclpy
from gym import wrappers
from tqdm import tqdm

from .qlearn import QLearn

from rclpy.logging import LoggingSeverity

from gym_envs.task_envs import TurtleBot3NavigationEnv


def _launch_custom_env(task_and_robot_environment_name: str, node: rclpy.node.Node):
    node.get_logger().warning("Env: {} will be imported".format(
        task_and_robot_environment_name))
    node.get_logger().warning("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
    env = gym.make(task_and_robot_environment_name, node=node)
    return env

def main():
    rclpy.init()
    node = rclpy.create_node('turtlebot3_world_qlearn')
    node.get_logger().set_level(LoggingSeverity.INFO)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = node.declare_parameter(
        'environment.task_and_robot_environment_name').value
    
    # env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    env: TurtleBot3NavigationEnv = _launch_custom_env(task_and_robot_environment_name, node)

    # Create the Gym environment
    node.get_logger().info("Gym environment done")
    node.get_logger().info("Starting Learning")

    # Set the logging system
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('miniproject')
    # outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")

    h_num_steps = np.ndarray(0)
    h_reward = np.ndarray(0)
    h_epsilon = np.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = node.declare_parameter("qlearning.alpha").value # Learning Rate
    assert isinstance(Alpha, float)

    Epsilon = node.declare_parameter("qlearning.epsilon").value
    assert isinstance(Epsilon, float)

    Gamma = node.declare_parameter("qlearning.gamma").value # Discount Factor
    assert isinstance(Gamma, float)

    epsilon_discount = node.declare_parameter("qlearning.epsilon_discount").value
    assert isinstance(epsilon_discount, float)

    nepisodes = node.declare_parameter("qlearning.nepisodes").value
    assert isinstance(nepisodes, int)

    nsteps = node.declare_parameter("qlearning.nsteps").value
    assert isinstance(nsteps, int)

    # world_lim_x_max = node.declare_parameter("/world/limits/x/max").value
    # world_lim_x_min = node.declare_parameter("/world/limits/x/min").value
    # world_lim_y_max = node.declare_parameter("/world/limits/y/max").value
    # world_lim_y_min = node.declare_parameter("/world/limits/y/min").value

    # obstacle_radius = node.declare_parameter("/world/obstacle_radius").value
    # obstacle_positions = node.declare_parameter("/world/obstacle_positions").value

    x = 0
    y = -1
    yaw = np.pi/2

    # Initialises the algorithm that we are going to use for learning
    qlearn = QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for ep in range(nepisodes):
        node.get_logger().info("############### START EPISODE=>" + str(ep))

        env.set_goal(x, y, yaw)

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        h_epsilon = np.append(h_epsilon, qlearn.epsilon)

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        # Move robot to new position. Must be done here and not _set_init_pose becuase the world is reset after that method is called.
        env.set_tb_state(0.0, 1.0, np.pi)
        observation = env.get_observation()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            node.get_logger().info("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            node.get_logger().debug("Next action is:%d" % (action))
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            node.get_logger().debug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            node.get_logger().debug("# state we were=>" + str(state))
            node.get_logger().debug("# action that we took=>" + str(action))
            node.get_logger().debug("# reward that action gave=>" + str(reward))
            node.get_logger().debug("# episode cumulated_reward=>" + str(cumulated_reward))
            node.get_logger().debug("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                node.get_logger().debug("NOT DONE")
                state = nextState
                if i == nsteps - 1:
                    node.get_logger().info("RAN OUT OF TIME STEPS")
                    h_num_steps = np.append(h_num_steps, i + 1)
            else:
                h_num_steps = np.append(h_num_steps, i + 1)     
                node.get_logger().info("DONE")
                break
            node.get_logger().debug("############### END Step=>" + str(i)+"/"+str(nsteps))
        
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        h_reward = np.append(h_reward, cumulated_reward)
        node.get_logger().error(("EP: " + str(ep + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    node.get_logger().info(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    node.get_logger().warning("Overall score: {:0.2f}".format(h_num_steps.mean()))

    plt.figure()
    plt.plot(range(nepisodes), h_num_steps)
    plt.xlabel("Episode")
    plt.ylabel("Num Steps")

    plt.figure()
    plt.plot(range(nepisodes), h_reward)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")

    plt.figure()
    plt.plot(range(nepisodes), h_epsilon)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.show()


    env.close()


if __name__ == '__main__':
    main()