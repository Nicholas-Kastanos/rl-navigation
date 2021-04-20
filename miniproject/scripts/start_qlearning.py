#!/usr/bin/env python

import gym
import numpy as np
import time
import qlearn
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
import gym_envs
from tqdm import tqdm
from functools import reduce

import matplotlib.pyplot as plt

def _launch_custom_env(task_and_robot_environment_name: str):
    rospy.logwarn("Env: {} will be imported".format(
        task_and_robot_environment_name))
    rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
    env = gym.make(task_and_robot_environment_name)
    return env


if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)

    # Init OpenAI_ROS ENV
    task_and_robot_environment_name = rospy.get_param(
        '/turtlebot3/task_and_robot_environment_name')
    
    # env = StartOpenAI_ROS_Environment(task_and_robot_environment_name)
    env = _launch_custom_env(task_and_robot_environment_name)

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

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
    Alpha = rospy.get_param("/turtlebot3/alpha") # Learning Rate
    Epsilon = rospy.get_param("/turtlebot3/epsilon") 
    Gamma = rospy.get_param("/turtlebot3/gamma") # Discount Factor
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")
    running_step = rospy.get_param("/turtlebot3/running_step")

    world_lim_x_max = rospy.get_param("/world/limits/x/max")
    world_lim_x_min = rospy.get_param("/world/limits/x/min")
    world_lim_y_max = rospy.get_param("/world/limits/y/max")
    world_lim_y_min = rospy.get_param("/world/limits/y/min")

    obstacle_radius = rospy.get_param("/world/obstacle_radius")
    obstacle_positions = rospy.get_param("/world/obstacle_positions")

    x = 0
    y = -1
    yaw = np.pi/2

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for ep in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(ep))

        env.set_goal(x, y, yaw)

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        h_epsilon = np.append(h_epsilon, qlearn.epsilon)

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        # Move robot to new position. Must be done here and not _set_init_pose becuase the world is reset after that method is called.
        env.set_tb_state(0, 1, np.pi)
        observation = env.get_observation()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in tqdm(range(nsteps)):
            rospy.logdebug("############### Start Step=>" + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logdebug("Next action is:%d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)

            rospy.logdebug(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logdebug("# state we were=>" + str(state))
            rospy.logdebug("# action that we took=>" + str(action))
            rospy.logdebug("# reward that action gave=>" + str(reward))
            rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
            rospy.logdebug("# State in which we will start next step=>" + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not (done):
                rospy.logdebug("NOT DONE")
                state = nextState
                if i == nsteps - 1:
                    rospy.logdebug("RAN OUT OF TIME STEPS")
                    h_num_steps = np.append(h_num_steps, i + 1)
            else:
                h_num_steps = np.append(h_num_steps, i + 1)     
                rospy.logdebug("DONE")
                break
            rospy.logdebug("############### END Step=>" + str(i)+"/"+str(nsteps))
        
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        h_reward = np.append(h_reward, cumulated_reward)
        rospy.logerr(("EP: " + str(ep + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    # print("Parameters: a="+str)
    rospy.logwarn("Overall score: {:0.2f}".format(h_num_steps.mean()))
    # rospy.logwarn("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

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