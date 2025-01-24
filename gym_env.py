"""
Contains the implementation of the quadrotor training environment
according to the Open AI *gym* specification. This will enable the use
of stable-baselines3 (referred to as sb3, henceforth). sb3 has RL
algorithms implemented in pytorch and simply required custom environments
to be of *gym* specification.
"""
from gym import spaces, Env
from dm_control import mujoco
from dm_control.utils.transformations import quat_to_euler
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np

XML_PATH = '../assets/old_quad.xml'

class QuadEnv(Env):
    """
    A Custom gym environment for the training of autonomous quadrotor
    system. This is intended to be used with the stable-baselines3
    package.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_file, seed=1001111, render_mode=None):
        # Set the render mode and the xml file path
        self.xml_file = xml_file
        self.physics = mujoco.Physics.from_xml_path(self.xml_file)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = "human" if render_mode is None else render_mode

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float64)

        self.rng = np.random.default_rng(seed)

    def render(self):
        return self.physics.render(height=720, width=1024, camera_id='fixed_camera')

    def get_observation(self):
        # Observation
        sensor_data = self.physics.data.sensordata.copy()
        sensor_data[6:] = sensor_data[6:]/3
        orientation = np.zeros(9)
        mjlib.mju_quat2Mat(orientation, self.physics.data.xquat[6].copy())
        # velocities = self.physics.data.qvel
        state = np.concatenate((sensor_data, orientation))#, velocities))
        return state

    def get_reward(self):
        # Penalise for orientation greater than pi/6
        atitude = self.physics.data.xquat[6].copy()
        euler = quat_to_euler(atitude)
        norm = np.round(np.sqrt(np.sum(np.square(euler))), 3)
        reward = -norm

        # Penalise if distance is less than 1
        sensor_array = self.physics.data.sensordata[6:].copy()
        sensor_array[:] = sensor_array/3
        for index, x in enumerate(sensor_array):
            if x<0.10:
                sensor_array[index] = -1
        reward += np.sum(sensor_array)

        return reward

    def step(self, action):
        self.physics.set_control(action)
        self.physics.step()
        done = True if self.physics.data.ncon > 0 or self.physics.data.time > 50 else False
        reward = self.get_reward()
        state = self.get_observation()
        return state, reward, done, {}

    def reset(self):
        with self.physics.reset_context():
            # x and y coordinates
            self.physics.data.qpos[:2] = self.rng.normal(0.0, 0.5, size=2)
            # z coordinate
            self.physics.data.qpos[2] = self.rng.uniform(0.5, 2)
            # initial velocities
            self.physics.data.qvel[:] = self.rng.normal(0.0, 1.0, size=6)
            # self.physics.data.qvel[3:] = rng.normal(0.0, 0.01, size=3)

        return self.get_observation()
