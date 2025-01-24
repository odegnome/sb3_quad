# Implementation of Deep Reinforcement Learning for Collision Prevention in Quadrotor - II

This is a continuation of my [dissertation project](https://github.com/odegnome/dcode) for building an autonomous quadrotor.
Difference from my original implementation is that this uses [sb3](https://github.com/DLR-RM/stable-baselines3)
algorithm to train and also, yaw axis acceleration is also a consideration in the reward.
The yaw axis reward was necessary because otherwise, the quadrotor was spinning about that
axis uncontrollably.

## Performance after training for 500k time-steps

https://github.com/odegnome/sb3_quad/blob/70e5ca4f52939106ca8389b1135dde0c84101bd8/500k.mp4
