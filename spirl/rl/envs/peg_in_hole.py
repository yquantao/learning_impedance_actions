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
from controller_manager_msgs.srv import SwitchController
from trajectory_msgs.msg import *
from geometry_msgs.msg import *
from panda_insertion.msg import *
import time

class PegInHoleEnv(BaseEnvironment, gym.Env):
    """Tiny wrapper for Peg-In-Hole tasks."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.goal = np.array([0.15, 0.32, 0.04])
        
        self.action_space = spaces.Box(np.array([-1]*8),np.array([1]*8)) # [x, y, z, theta_z, stifness k_x, k_y, k_z, k_theta_z]
        self.observation_space = spaces.Box(np.array([-5.0]*22),np.array([5.0]*22)) # [7 joint positions, 7 joint velocities, 7 cartesian pose, contact force]
        self.reward = 0
        self.update_obs = False

        rospy.init_node('DRL_node', anonymous=True)
        self.equilibrium_pose_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=None)
        self.desired_stifffness_pub = rospy.Publisher('/desired_stiffness', TwistStamped, queue_size=None)
        self.state_sub = rospy.Subscriber("/insertion_spirl/state", StateMsg, self._next_observation, queue_size=1)
        self.rate = rospy.Rate(5)

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': None,   # name of openai/gym environment
            'reward_norm': 1.,  # reward normalization factor
            'punish_reward': -100,   # reward used when action leads to simulation crash
            'unwrap_time': True,    # removes time limit wrapper from envs so that done is not set on timeout
        })

        return super()._default_hparams().overwrite(default_dict)

    def reset_sim(self):
        print("***reset function***")

        # move to initial pose using position joint trajectory controller
        switch_controller = rospy.ServiceProxy('/panda/controller_manager/switch_controller', SwitchController)
        #resp = switch_controller({'effort_joint_trajectory_controller'},{'impedance_controller'},2,True,0.1)
        #init_trajectory_pub = rospy.Publisher('/effort_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        resp = switch_controller({'position_joint_trajectory_controller'},{'cartesian_impedance_mod_controller'},2)
        init_trajectory_pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        joints = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        time.sleep(1.0)
        init_trajectory_pub.publish(JointTrajectory(joint_names=joints, points=[
                JointTrajectoryPoint(positions=[0.81, -0.78, -0.17, -2.35, -0.12, 1.60, 0.75], time_from_start=rospy.Duration(4.0))]))

        time.sleep(5.0)

        # switch back to impedance controller
        #resp = switch_controller({'impedance_controller'},{'effort_joint_trajectory_controller'},2,True,0.1)
        resp = switch_controller({'cartesian_impedance_mod_controller'},{'position_joint_trajectory_controller'},2)

        while not self.update_obs:
            self.rate.sleep()

        return self._wrap_observation(self.obs)

    def reset(self):
        print("***reset function***")

        # move to initial pose
        time.sleep(1.0)
        desired_stiffness = TwistStamped()
        desired_stiffness.twist.linear.x = 400
        desired_stiffness.twist.linear.y = 400
        desired_stiffness.twist.linear.z = 400
        desired_stiffness.twist.angular.x = 60
        desired_stiffness.twist.angular.y = 60
        desired_stiffness.twist.angular.z = 60

        start_point = self.translation
        step_size = 0.1/100

        for i in range(100):
            reset_pose = PoseStamped()
            reset_pose.header.frame_id = "panda_link0"
            reset_pose.pose.position.x = start_point[0][0]                   
            reset_pose.pose.position.y = start_point[0][1]                  
            reset_pose.pose.position.z = start_point[0][2]+i*step_size
            reset_pose.pose.orientation.x = -0.999726
            reset_pose.pose.orientation.y = 0.014363
            reset_pose.pose.orientation.z = -0.0171982
            reset_pose.pose.orientation.w = 0.00678687

            self.equilibrium_pose_pub.publish(reset_pose)
            self.desired_stifffness_pub.publish(desired_stiffness)
        time.sleep(1.0)

        start_point = self.translation
        step_x = (0.275-self.translation[0][0])/200
        step_y = (0.170-self.translation[0][1])/200
        step_z = (0.289-self.translation[0][2])/200
        for i in range(200):
            reset_pose = PoseStamped()
            reset_pose.header.frame_id = "panda_link0"
            reset_pose.pose.position.x = start_point[0][0]+i*step_x
            reset_pose.pose.position.y = start_point[0][1]+i*step_y
            reset_pose.pose.position.z = start_point[0][2]+i*step_z
            reset_pose.pose.orientation.x = -0.999726
            reset_pose.pose.orientation.y = 0.014363
            reset_pose.pose.orientation.z = -0.0171982
            reset_pose.pose.orientation.w = 0.00678687

            self.equilibrium_pose_pub.publish(reset_pose)
            self.desired_stifffness_pub.publish(desired_stiffness)
        time.sleep(0.1)

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
        desired_stiffness.twist.angular.x = 60
        desired_stiffness.twist.angular.y = 60
        desired_stiffness.twist.angular.z = action[7]

        # publish action messages
        print(equilibrium_pose)
        self.equilibrium_pose_pub.publish(equilibrium_pose)
        self.desired_stifffness_pub.publish(desired_stiffness)
        time.sleep(2.0)

        # wait for next obs
        self.update_obs = False
        while not self.update_obs:
            self.rate.sleep()

        self.reward, done = self.calculate_reward()
        print("reward: ", self.reward)
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
        dist = np.abs(self.translation[0][2]-self.goal[2])

        if mode == 0:
            reward = -dist
            if dist < 0.01 and self.translation[2] < 0.045:
                reward = 1.0
                done = True
        elif mode == 1:
            if dist < 0.01 and self.translation[2] < 0.045:
                reward = 1.0
                done = True

        return reward, done
    