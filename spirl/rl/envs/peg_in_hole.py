import numpy as np
import rospy
import gym
from gym import spaces

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv
from spirl.rl.components.environment import BaseEnvironment
from panda_insertion.msg import StateMsg
from panda_insertion.msg import ActionMsg

from controller_manager_msgs.srv import *
from trajectory_msgs.msg import *

class PegInHoleEnv(BaseEnvironment, gym.Env):
    """Tiny wrapper for Peg-In-Hole tasks."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.goal = np.array([0.1, 0.1, 0])
        
        self.action_space = spaces.Box(np.array([-1]*8),np.array([1]*8)) # [x, y, z, theta_z, stifness k_x, k_y, k_z, k_theta_z]
        self.observation_space = spaces.Box(np.array([-5.0]*22),np.array([5.0]*22)) # [7 joint positions, 7 joint velocities, 7 cartesian pose, contact force]
        self.reward = 0
        self.update_obs = False

        rospy.init_node('DRL_node', anonymous=True)
        self.action_pub = rospy.Publisher('/insertion_spirl/action', ActionMsg)
        self.state_sub = rospy.Subscriber("/insertion_spirl/state", StateMsg, self._next_observation, queue_size=1)
        self.rate = rospy.Rate(10)

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': None,   # name of openai/gym environment
            'reward_norm': 1.,  # reward normalization factor
            'punish_reward': -100,   # reward used when action leads to simulation crash
            'unwrap_time': True,    # removes time limit wrapper from envs so that done is not set on timeout
        })

        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        # move to initial pose using position joint trajectory controller
        switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        resp = switch_controller({'position_joint_trajectory_controller'},{'impedance_controller'},2,True,0.1)
        init_trajectory_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        joints = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        init_trajectory_pub.publish(JointTrajectory(joint_names=joints, points=[
                JointTrajectoryPoint(positions=[0.81, -0.78, -0.17, -2.35, -0.12, 1.60, 0.75], time_from_start=rospy.Duration(4.0))]))

        # switch back to impedance controller
        resp = switch_controller({'impedance_controller'},{'position_joint_trajectory_controller'},2,True,0.1)

        while not self.fresh:
            self.rate.sleep()

        return self._wrap_observation(self.obs)

    def step(self, action):
        #obs, reward, done, info = self._env.step(action)
        self.pub.publish(action)

        # wait for next obs
        self.update_obs = False
        while not self.update_obs:
            self.rate.sleep()

        self.reward, done = self.calculate_reward()
        info = {}

        return self._wrap_observation(self.obs), self.reward, done, info

    def _next_observation(self, msg):
        self.q = np.array(msg.q)
        self.dq = np.array(msg.dq)
        self.pose = np.array(msg.pose)
        self.f_ext = np.array(msg.f_ext)

        # next observation
        self.obs = np.array(msg)
        self.update_obs = True

    def calculate_reward(self, mode=0):
        # mode=0: dense reward; model=1: sparse reward
        reward = 0
        if mode == 0:
            reward = -10.0*np.linalg.norm(self.pose[0:3]-self.goal)
            done = False
        else:
            if self.f_ext < 0.9:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False

        return reward, done
    