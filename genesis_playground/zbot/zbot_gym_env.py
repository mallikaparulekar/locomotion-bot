import gym
from gym import spaces
import numpy as np
import torch  # Make sure to import torch

class ZBotGymEnv(gym.Env):
    def __init__(self, zbot_env):
        super(ZBotGymEnv, self).__init__()
        self.zbot_env = zbot_env

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.zbot_env.num_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.zbot_env.env_cfg["clip_actions"], 
            high=self.zbot_env.env_cfg["clip_actions"], 
            shape=(self.zbot_env.num_actions,), dtype=np.float32
        )

    def _to_cpu(self, obs):
        """Recursively convert torch tensors to CPU numpy arrays."""
        if torch.is_tensor(obs):
            return obs.cpu().numpy()
        elif isinstance(obs, dict):
            return {k: self._to_cpu(v) for k, v in obs.items()}
        elif isinstance(obs, list):
            return [self._to_cpu(o) for o in obs]
        return obs

    def reset(self):
        # Reset the environment
        obs, _ = self.zbot_env.reset()
        obs = self._to_cpu(obs)
        return obs

    def step(self, action):
        # Take a step in the environment
        obs, _, reward, done, _ = self.zbot_env.step(action)
        obs = self._to_cpu(obs)
        return obs, reward, done, {}

    def render(self, mode='human'):
        # If you want to visualize the environment, implement this
        # You can visualize using self.zbot_env.scene.render() or any custom method.
        pass