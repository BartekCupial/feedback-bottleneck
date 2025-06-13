import gym
import pygame
from gym.utils.play import display_arr


class PyGameWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.current_seed = None

        self.video_size = (512, 512)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.fps = 30

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        if mode == "human":
            rendered = self.env.render(size=self.video_size)

            if self.video_size is None:
                self.video_size = [rendered.shape[1], rendered.shape[0]]

            if self.screen is None:
                self.screen = pygame.display.set_mode(self.video_size)

            display_arr(self.screen, rendered, transpose=True, video_size=self.video_size)

            pygame.display.flip()
            self.clock.tick(self.fps)

        elif mode == "rgb_array":
            rendered = self.env.render()
            return rendered

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

        return super().close()
