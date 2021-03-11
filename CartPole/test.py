import gym
import numpy as np
from RLmodel import build_model
from RLagent import build_agent
from tensorflow.keras.optimizers import Adam

env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3))

dqn.load_weights('weights/dqn_weights.h5f')
_ = dqn.test(env, nb_episodes=5, visualize=True)