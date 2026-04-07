import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, grid_size=10, max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=max(grid_size - 1, 1), shape=(7,), dtype=np.float32)

        self.agent_pos = None
        self.target_pos = None
        self.carrying = False
        self.steps = 0
        self.package_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.package_pos = self.np_random.integers(0, self.grid_size, size=2).astype(np.int32)
        self.target_pos = self.np_random.integers(0, self.grid_size, size=2).astype(np.int32)

        while np.array_equal(self.package_pos, self.agent_pos):
            self.package_pos = self.np_random.integers(0, self.grid_size, size=2).astype(np.int32)
        while np.array_equal(self.target_pos, self.agent_pos):
            self.target_pos = self.np_random.integers(0, self.grid_size, size=2).astype(np.int32)

        self.carrying = False
        self.steps = 0
        return self._get_obs(), {"grid_size": self.grid_size}

    def step(self, action):
        self.steps += 1
        reward = -0.01
        terminated = False
        truncated = self.steps >= self.max_steps

        if action == 0: self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 1: self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 2: self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3: self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        elif action == 4:
            if not self.carrying and np.array_equal(self.agent_pos, self.package_pos):
                self.carrying = True; reward += 0.5
            else: reward -= 0.5
        elif action == 5:
            if self.carrying and np.array_equal(self.agent_pos, self.target_pos):
                self.carrying = False; reward += 10.0; terminated = True
            else: reward -= 0.5

        info = {"steps": self.steps, "carrying": self.carrying}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array([
            float(self.agent_pos[0]), float(self.agent_pos[1]),
            float(self.target_pos[0]), float(self.target_pos[1]),
            float(self.carrying),
            float(self.target_pos[0] - self.agent_pos[0]),
            float(self.target_pos[1] - self.agent_pos[1])
        ], dtype=np.float32)

def make_env(render_mode=None, **kwargs):
    def _init():
        return DeliveryEnv(render_mode=render_mode, **kwargs)
    return _init
