# Implementation of Deep Reinforcement Learning for Collision Prevention in Quadrotor - II

This is a continuation of my [dissertation project](https://github.com/odegnome/dcode) for building an autonomous quadrotor.
Difference from my original implementation is that this uses [sb3](https://github.com/DLR-RM/stable-baselines3)
algorithm to train and also, yaw axis acceleration is also a consideration in the reward.
The yaw axis reward was necessary because otherwise, the quadrotor was spinning about that
axis uncontrollably.

## Performance after training for 500k time-steps

There is a glitch in the video below due to incorrect use of ffmpeg for converting images to mp4. Please ignore!

https://github.com/user-attachments/assets/e1aca736-3fb0-4e03-85ce-7df2cd698cc4

