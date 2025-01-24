from gym_env import QuadEnv
from dm_control import mujoco
from stable_baselines3 import DDPG
from PIL import Image
import os

# to create a video
# ffmpeg -framerate 30 -i 'simulation/frame%03d.jpeg' video.mpg
# ffmpeg -framerate 30 -pattern_type glob -i 'simulation/*.jpeg' video.mpg

framerate = 30
duration = 2

env = QuadEnv("/Users/rishabh/project/sb3_quad/old_quad.xml", seed=88543)
model = DDPG.load("policy/2500k", env=env)
def simulate():
    count = 0
    counter = 0
    total_reward = 0
    obs = env.reset()
    # with physics.reset_context():
    #     physics.set_control(controls[index][0:4])
    #     physics.data.qvel[1] = controls[index][4]
    #     physics.data.qvel[2] = controls[index][5]
    done = False
    print(env.physics.data.time, end=' | ')
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if counter < env.physics.data.time * framerate:
            pixels = env.physics.render(height=720,width=1024,camera_id='fixed_camera')
            im = Image.fromarray(pixels)
            im.save(f"simulation/frame{count:05}.jpeg")
            count += 1
            counter += 1
    print(env.physics.data.time)
    print(f'{total_reward=}')

simulate()
