from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from PaddleEnv import PaddleEnv

model = DQN.load("model.h5")
env = DummyVecEnv([lambda: PaddleEnv()])
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _ = env.step(action)