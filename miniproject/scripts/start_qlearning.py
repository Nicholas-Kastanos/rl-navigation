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

from gym_envs.task_envs import TurtleBot3NavigationEnv

import matplotlib.pyplot as plt
import pickle
import os

def _launch_custom_env(task_and_robot_environment_name: str):
    rospy.logwarn("Env: {} will be imported".format(
        task_and_robot_environment_name))
    rospy.logwarn("Register of Task Env went OK, lets make the env..."+str(task_and_robot_environment_name))
    env = gym.make(task_and_robot_environment_name)
    return env

def _save_q(ep, q, h_num_steps, h_reward, h_epsilon):
    chkp_dir = os.path.join('/', 'home', 'nicholas', 'project_cws', 'src', 'miniproject', 'checkpoints')
    os.makedirs(chkp_dir, exist_ok=True)
    with open(os.path.join(chkp_dir, 'chkp.pickle'), 'wb') as handle:
        pickle.dump(q, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(chkp_dir, 'ep.pickle'), 'wb') as handle:
        pickle.dump((ep, h_num_steps, h_reward, h_epsilon), handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_q():
    rospy.logerr("Looking for Checkpoint...")
    chkp_dir = os.path.join('/', 'home', 'nicholas', 'project_cws', 'src', 'miniproject', 'checkpoints')
    if not os.path.isfile(os.path.join(chkp_dir, 'chkp.pickle')):
        rospy.logerr("No Checkpoint Found")
        return 0, {}, np.ndarray(0), np.ndarray(0), np.ndarray(0)
    rospy.logerr("Found")
    with open(os.path.join(chkp_dir, 'chkp.pickle'), 'rb') as chkp:
        q =  pickle.load(chkp)
        with open(os.path.join(chkp_dir, 'ep.pickle'), 'rb') as info:
            ep, h_num_steps, h_reward, h_epsilon = pickle.load(info)
    
            return ep, q, h_num_steps, h_reward, h_epsilon

if __name__ == '__main__':

    rospy.init_node('turtlebot3_world_qlearn', anonymous=True, log_level=rospy.WARN)
    task_and_robot_environment_name = rospy.get_param('/turtlebot3/task_and_robot_environment_name')
    env: TurtleBot3NavigationEnv = _launch_custom_env(task_and_robot_environment_name)

    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    Alpha = rospy.get_param("/turtlebot3/alpha") # Learning Rate
    Epsilon = rospy.get_param("/turtlebot3/epsilon") 
    Gamma = rospy.get_param("/turtlebot3/gamma") # Discount Factor
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")
    running_step = rospy.get_param("/turtlebot3/running_step")

    x = 1
    y = 0
    yaw = np.pi/2

    Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    start_ep, q, h_num_steps, h_reward, h_epsilon  = _load_q()
    qlearn.q = q

    env.episode_num = start_ep

    Starts the main training loop: the one about the episodes to do
    for ep in range(start_ep, nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(ep))

        env.set_goal(x, y, yaw)

        cumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon = epsilon_discount**ep
        h_epsilon = np.append(h_epsilon, qlearn.epsilon)

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        # Move robot to new position. Must be done here and not _set_init_pose becuase the world is reset after that method is called.
        env.set_tb_state(-1, 0, 0)
        observation = env.get_observation()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        # for i in tqdm(range(nsteps)):
        i = 0
        with tqdm() as pbar:
            while not done and not rospy.is_shutdown():
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
                i += 1
                pbar.update()
            
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        h_reward = np.append(h_reward, cumulated_reward)
        rospy.logerr(("EP: " + str(ep + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

        _save_q(ep, qlearn.q, h_num_steps, h_reward, h_epsilon)

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    # print("Parameters: a="+str)
    rospy.logwarn("Overall score: {:0.2f}".format(h_num_steps.mean()))
    rospy.logwarn("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # plt.figure()
    # plt.plot(range(nepisodes), h_num_steps)
    # plt.xlabel("Episode")
    # plt.ylabel("Num Steps")

    plt.figure()
    plt.plot(range(nepisodes), h_reward)
    plt.plot(h_reward)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.tight_layout()

    plt.figure()
    plt.plot(range(nepisodes), h_epsilon)
    plt.plot(h_epsilon)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.show()


    env.close()