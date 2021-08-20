import numpy as np
import rospy
import gym
from gym import spaces
from rospy.topics import Publisher

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv
from spirl.rl.components.environment import BaseEnvironment

#from tf2.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation
from controller_manager_msgs.srv import *
from trajectory_msgs.msg import *
from geometry_msgs.msg import *
from panda_insertion.msg import *
import time

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
        self.equilibrium_pose_pub = rospy.Publisher('/impedance_controller/equilibrium_pose', PoseStamped, queue_size=None)
        self.desired_stifffness_pub = rospy.Publisher('/impedance_controller/desired_stiffness', TwistStamped, queue_size=None)
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
        print("***reset function***")
        # move to initial pose using position joint trajectory controller
        switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        resp = switch_controller({'effort_joint_trajectory_controller'},{'impedance_controller'},2,True,0.1)
        init_trajectory_pub = rospy.Publisher('/effort_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        joints = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        time.sleep(1.0)
        init_trajectory_pub.publish(JointTrajectory(joint_names=joints, points=[
                JointTrajectoryPoint(positions=[0.81, -0.78, -0.17, -2.35, -0.12, 1.60, 0.75], time_from_start=rospy.Duration(4.0))]))

        time.sleep(5.0)

        # switch back to impedance controller
        resp = switch_controller({'impedance_controller'},{'effort_joint_trajectory_controller'},2,True,0.1)

        while not self.update_obs:
            self.rate.sleep()

        return self._wrap_observation(self.obs)

    def step(self, action):
        # publish equilibrium pose and desired stiffness
        action = action.tolist()
        
        # equilibrium pose
        equilibrium_pose = PoseStamped()
        equilibrium_pose.pose.position.x = action[0]
        equilibrium_pose.pose.position.y = action[1]
        equilibrium_pose.pose.position.z = action[2]
        rot = Rotation.from_euler('xyz', [np.pi, 0, action[3]], degrees=False)
        quaternion = rot.as_quat()
        equilibrium_pose.pose.orientation.x = quaternion[0]
        equilibrium_pose.pose.orientation.y = quaternion[1]
        equilibrium_pose.pose.orientation.z = quaternion[2]
        equilibrium_pose.pose.orientation.w = quaternion[3]

        # desired stiffness
        desired_stiffness = TwistStamped()
        desired_stiffness.twist.linear.x = action[4]
        desired_stiffness.twist.linear.y = action[5]
        desired_stiffness.twist.linear.z = action[6]
        desired_stiffness.twist.angular.x = 80
        desired_stiffness.twist.angular.y = 80
        desired_stiffness.twist.angular.z = action[7]

        # publish action messages
        self.equilibrium_pose_pub.publish(equilibrium_pose)
        self.desired_stifffness_pub.publish(desired_stiffness)

        # wait for next obs
        self.update_obs = False
        while not self.update_obs:
            self.rate.sleep()

        self.reward, done = self.calculate_reward()
        info = {}

        return self._wrap_observation(self.obs), self.reward, np.array(done), info

    def _next_observation(self, msg):
        self.q = np.array(msg.q).reshape(1,-1)
        self.dq = np.array(msg.dq).reshape(1,-1)
        self.translation = np.array(msg.ee_translation).reshape(1,-1)
        self.theta_z = np.array(msg.theta_z).reshape(1,-1)
        self.f_ext = np.array(msg.f_ext).reshape(1,-1)

        # next observation
        #self.obs = np.array(msg)
        obs = np.concatenate([self.q, self.dq, self.translation, self.theta_z, self.f_ext], axis=1)
        self.obs = obs.reshape(-1)
        self.update_obs = True

    def calculate_reward(self, mode=0):
        # mode=0: dense reward; model=1: sparse reward
        reward = 0
        done = False
        
        dist = np.linalg.norm(self.translation[0:3]-self.goal)

        if mode == 0:
            reward = -dist
            if dist < 0.01:
                done = True
        elif mode == 1:
            if self.dist < 0.01:
                reward = 1.0
                done = True

        return reward, done
    