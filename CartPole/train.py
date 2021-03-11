import gym
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

env = gym.make('CartPole-v1')

env = DummyVecEnv([lambda: env])
model = DQN('MlpPolicy', env, verbose=1,
            tensorboard_log="./dqn_cartpole_tensorboard/")

model.learn(total_timesteps=100000)

result = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(result)
env.close()

model.save('cartpole_model')