import gymnasium as gym
import numpy as np
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
from collections import deque
from omegaconf import OmegaConf, open_dict

from tdmpc2_jax.envs.wrappers.pixels import PixelWrapper


# TODO: Remove [0] indexing by directly using RaisimGymEnv
class Raisim(gym.Env):
    render_mode = "rgb_array"

    def __init__(self, seed: int, cfg: dict):
        self.cfg = cfg
        self.seed = seed
        self.env = VecEnv(RaisimGymEnv(cfg.resource_path, OmegaConf.to_yaml(self.cfg)))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.num_obs,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.env.num_acts,), dtype=np.float32
        )
        if self.cfg.visualize_seed == self.seed:
            self.env.turn_on_visualization()

    def reset(
        self, *, seed: int | None = None, options: dict[str, any] | None = None
    ) -> tuple[np.ndarray, dict[str, any]]:
        self.env.curriculum_callback()
        self.env.reset()
        obs = self.env.observe().astype(np.float32)[0]
        return obs, {}

    def render(self) -> np.ndarray:
        im = (
            np.nan_to_num(self.env.depth_image()[0], nan=255)
            .clip(0, 255)
            .transpose((1, 0))
        )
        return np.expand_dims(im, 0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, any]]:
        rewards, terms = self.env.step(action.reshape((1, -1)))
        reward = rewards[0]
        term = terms[0]
        obs = self.env.observe().astype(np.float32)[0]
        info = {}
        info["success"] = not term
        return obs, reward, term, False, info

    def close(self):
        if self.cfg.visualize_seed == self.seed:
            self.env.turn_off_visualization()
        self.env.close()


def make_raisim_env(id: str, seed: int, cfg: dict, obs_type: str):
    env = Raisim(seed, cfg.raisim)
    env = gym.wrappers.TimeLimit(env, cfg.raisim.max_time // cfg.raisim.control_dt)
    if obs_type == "depth":
        env = PixelWrapper(env, num_channels=1)
    if obs_type == "multimodal":
        env = PixelWrapper(env, num_channels=1, pixels_only=False)
    return env
