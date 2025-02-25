# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import torch  # Make sure to import torch

from stable_baselines3.common.vec_env import VecEnv


class ZBotVecEnv(VecEnv):
    def __init__(self, zbot_env):
        self.zbot_env = zbot_env
        self.num_envs = zbot_env.num_envs
        # Define observation and action spaces as before
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.zbot_env.num_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.zbot_env.env_cfg["clip_actions"],
            high=self.zbot_env.env_cfg["clip_actions"],
            shape=(self.zbot_env.num_actions,), dtype=np.float32
        )
    
    def reset(self):
        obs, _ = self.zbot_env.reset()
        obs = self._to_cpu(obs)
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        return obs  # shape: (num_envs, obs_dim)
    
    def step_async(self, actions):
        self._actions = actions
    
    def step_wait(self):
        return self.step(self._actions)
    
    def step(self, actions):
        # Unpack the five values; ignore the second one
        obs, _, reward, done, info = self.zbot_env.step(actions)
        
        # Convert torch tensors to numpy arrays if needed
        obs = self._to_cpu(obs)
        reward = self._to_cpu(reward)
        done = self._to_cpu(done)
        
        # Ensure info is a dictionary
        if not isinstance(info, dict):
            info = {}
        
        # If the observation is unbatched, add a batch dimension
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
            reward = np.array([reward])
            done = np.array([done])
        
        # Wrap info in a list (one per environment)
        infos = [info.copy() for _ in range(self.num_envs)]
        
        return obs, reward, done, infos




    
    def close(self):
        pass
    
    def _to_cpu(self, obs):
        if torch.is_tensor(obs):
            return obs.cpu().numpy()
        elif isinstance(obs, dict):
            return {k: self._to_cpu(v) for k, v in obs.items()}
        elif isinstance(obs, list):
            return [self._to_cpu(o) for o in obs]
        return obs

    # --- Minimal implementations of the abstract methods ---
    def get_attr(self, attr_name, indices=None):
        # Return the attribute from the underlying environment.
        # Here we assume a single underlying environment.
        return [getattr(self.zbot_env, attr_name)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.zbot_env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.zbot_env, method_name)
        return [method(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        # For simplicity, say that our env is not wrapped.
        return [False for _ in range(self.num_envs)]


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
        obs, _ = self.zbot_env.reset()
        obs = self._to_cpu(obs)
        print(f"DEBUG: Reset obs shape = {obs.shape}")
        # If the observation is a 1D array (i.e. single env), make it 2D.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        return obs  # Return a numpy array of shape (num_envs, obs_dim)

    def step(self, action):
        obs, _, reward, done, _ = self.zbot_env.step(action)
        obs = self._to_cpu(obs)
        reward = self._to_cpu(reward)
        done = self._to_cpu(done)
        print(f"DEBUG: Step obs shape = {obs.shape}")
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
            reward = np.array([reward])
            done = np.array([done])
        return obs, reward, done, [{}] * obs.shape[0]





    def render(self, mode='human'):
        # If you want to visualize the environment, implement this
        # You can visualize using self.zbot_env.scene.render() or any custom method.
        pass