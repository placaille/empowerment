import numpy as np
import gym

from gym import spaces


class StaticEnv(gym.Env):

    def reset(self):
        raise NotImplemented

    def step(self, action):
        raise NotImplemented


class DiscreteStaticEnv(StaticEnv):
    def __init__(self):

        # convert map into np array
        convert_fn = lambda c: self.wall if c == self.wall_str else self.free
        self.grid = np.array([list(map(convert_fn, l)) for l in self.map.splitlines()])

        # define actions
        self.actions = {
            'UP': 0,
            'DOWN': 1,
            'RIGHT': 2,
            'LEFT': 3
        }
        self.directions = np.array([
            [-1, 0],  # 0-UP
            [1, 0],  # 1-DOWN
            [0, 1],  # 2-RIGHT
            [0, -1],  # 3-LEFT
        ])

        # define appropriate action and observation spaces
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete((self.grid == self.free).sum())
        self.free_states = list(range(self.observation_space.n))

        # setup pos <-> state
        self.pos_to_state = {}
        self.state_to_pos = {}
        state_id = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == self.free:
                    self.pos_to_state[(i, j)] = state_id
                    self.state_to_pos[state_id] = np.array([i, j])
                    state_id += 1

    def reset(self, state_id=None):
        if state_id:
            assert self.observation_space.contains(state_id), 'ValueError: {}'.format(state_id)
            self.state = state_id
        else:
            self.state = self.observation_space.sample()
        self.pos = self.state_to_pos[self.state]
        return self._convert_state_to_onehot(self.state)

    def step(self, action):
        assert self.action_space.contains(action), 'ValueError: {}'.format(action)
        new_pos = tuple(self.pos + self.directions[action])

        # test if new pos is a free position, otherwise don't move
        if self.grid[new_pos] == self.free:
            self.pos = np.array(new_pos)
            self.state = self.pos_to_state[new_pos]
        return self._convert_state_to_onehot(self.state)

    def __str__(self):
        return self.map

    def _convert_state_to_onehot(self, state):
        onehot = np.zeros(self.observation_space.n)
        onehot[state] = 1.0
        return onehot

class TwoRoomVarInfo(DiscreteStaticEnv):
    def __init__(self):
        """
        Simple two room environment from var-info paper (non-symmetric)

        W:      Wall (not attainable)
        else:   Attainable tile
        """
        self.map = """\
WWWWWWWWWWWWWWWWWWWW
W        W         W
W        W         W
W        W         W
W        W         W
W        W         W
W        W         W
W        W         W
W                  W
W                  W
W                  W
W                  W
W                  W
W        W         W
W        W         W
W        W         W
W        W         W
W        W         W
W        W         W
WWWWWWWWWWWWWWWWWWWW
"""
        # specific to env
        self.wall, self.wall_str = 0, 'W'
        self.free, self.free_str = 1, ' '

        # init env
        super(TwoRoomVarInfo, self).__init__()


class CrossRoomVarInfo(DiscreteStaticEnv):
    def __init__(self):
        """
        Simple cross room environment from var-info paper (non-symmetric)

        W:      Wall (not attainable)
        else:   Attainable tile
        """
        self.map = """\
WWWWWWWWWWWWWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
W                  W
W                  W
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWW  WWWWWWWWW
WWWWWWWWWWWWWWWWWWWW
"""
        # specific to env
        self.wall, self.wall_str = 0, 'W'
        self.free, self.free_str = 1, ' '

        # init env
        super(CrossRoomVarInfo, self).__init__()


class RoomPlus2CorridVarInfo(DiscreteStaticEnv):
    def __init__(self):
        """
        Simple room with two corridors environment from var-info paper (non-symmetric)

        W:      Wall (not attainable)
        else:   Attainable tile
        """
        self.map = """\
WWWWWWWWW
W W     W
W W WWWWW
W W     W
W W     W
W       W
W W     W
WWWWWWWWW
"""
        # specific to env
        self.wall, self.wall_str = 0, 'W'
        self.free, self.free_str = 1, ' '

        # init env
        super(RoomPlus2CorridVarInfo, self).__init__()
