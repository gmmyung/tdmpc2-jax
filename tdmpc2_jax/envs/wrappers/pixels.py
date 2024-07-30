from collections import deque

import gymnasium as gym
import numpy as np


class PixelWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with DMControl environments.
    """

    def __init__(
        self, env, num_frames=3, num_channels=3, pixels_only=True, render_size=64
    ):
        super().__init__(env)
        self.env = env
        self._pixels_only = pixels_only
        if self._pixels_only:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(num_frames * num_channels, render_size, render_size),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": env.observation_space,
                    "pixels": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(num_frames * num_channels, render_size, render_size),
                        dtype=np.float32,
                    ),
                }
            )

        self._frames = deque([], maxlen=num_frames)
        self._render_size = render_size

    def _get_pixels(self):
        frame = self.env.render()
        self._frames.append(frame)
        return np.concatenate(self._frames)

    def reset(self, seed=None, options=None):
        state, info = self.env.reset()
        for _ in range(self._frames.maxlen):
            pixel = self._get_pixels()
        if self._pixels_only:
            obs = pixel
        else:
            obs = {"state": state, "pixels": pixel}
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        pixel = self._get_pixels()
        if self._pixels_only:
            obs = pixel
        else:
            obs = {"state": obs, "pixels": pixel}
        return obs, reward, term, trunc, info
