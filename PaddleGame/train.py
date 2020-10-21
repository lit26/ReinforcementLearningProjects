from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from PaddleEnv import PaddleEnv

env = DummyVecEnv([lambda: PaddleEnv()])

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000, log_interval=10)
model.save("model.h5")
