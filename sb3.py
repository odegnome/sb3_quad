from gym_env import QuadEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
import numpy as np
import time
from datetime import timedelta

env = QuadEnv("/Users/rishabh/project/sb3_quad/old_quad.xml")
check_env(env)
print("Env is valid for SB3")
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1*np.ones(n_actions))
print("Starting training")
# model = DDPG(policy="MlpPolicy", env=env, action_noise=action_noise, verbose=1)
# model.learn(100000, log_interval=10)
# model.save("sb3_policy_100000")
ts = 500_000
print(f"Starting training for {ts=}")
model = DDPG.load("policy/2500k",env=env)
start = time.time()
model.learn(ts, log_interval=100)
end = time.time()
print(f"Elapsed: {timedelta(seconds=end-start)}")
model.save("policy/3000k")
