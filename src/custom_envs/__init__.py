from custom_envs.static_envs import StaticEnv
import gym

gym.envs.register(
     id='StaticEnv-v0',
     entry_point='custom_envs.static_envs:StaticEnv',
)
