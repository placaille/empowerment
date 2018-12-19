# from custom_envs.static_envs import StaticEnv, TwoRoomVarInfo, CrossRoomVarInfo
import gym

gym.envs.register(
    id='StaticEnv-v0',
    entry_point='custom_envs.static_envs:StaticEnv',
)

gym.envs.register(
    id='TwoRoom-v0',
    entry_point='custom_envs.static_envs:TwoRoomVarInfo',
)

gym.envs.register(
    id='CrossRoom-v0',
    entry_point='custom_envs.static_envs:CrossRoomVarInfo',
)

gym.envs.register(
    id='RoomPlus2Corrid-v0',
    entry_point='custom_envs.static_envs:RoomPlus2CorridVarInfo',
)
