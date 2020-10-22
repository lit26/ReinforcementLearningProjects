from Paddle import Paddle
import numpy as np
import gym
from gym import spaces

class PaddleEnv(gym.Env):

    def __init__(self):
        super(PaddleEnv, self).__init__()
        self.game = Paddle()

        # Actions: left right hold
        self.action_space = spaces.Discrete(3)

        # Observation: paddle x coordinates, ball location and ball direction
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the state of the environment
        p_xcor = self.game.paddle.xcor()
        b_xcor = self.game.ball.xcor()
        b_ycor = self.game.ball.ycor()

        # normalize the state to -1 to 1
        state = [p_xcor/300,
                b_xcor/300,
                b_ycor/300,
                 (p_xcor-b_xcor)/600,
                self.game.ball.dx/4,
                self.game.ball.dy/3]
        state = np.reshape(state, (1, 6))
        return state

    def reset(self):
        # Reset the state of the environment to an initial state
        self.game.reset()
        return self._next_observation()

    def step(self, action):
        # Execute one time step within the environment
        reward = 0
        if action == 0:
            self.game.movement(action='left')
        elif action == 2:
            self.game.movement(action='right')

        hit, done = self.game.run_frame()
        if (hit):
            reward += 5
        elif (done):
            reward -= 5
        state = self._next_observation()
        return state, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass
