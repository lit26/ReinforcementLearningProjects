import gym
from stable_baselines import DQN

env = gym.make('CartPole-v1')

model = DQN.load("cartpole_model")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break