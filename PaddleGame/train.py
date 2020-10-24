from DQN import DQN
from PaddleEnv import PaddleEnv
import pickle

class PaddleModel:
    def __init__(self, save_episode=20, max_timestamp=100000, init=True):
        self.max_timestamp = max_timestamp
        self.env = PaddleEnv()
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[1]
        if init:
            self.agent = DQN(self.action_space, self.state_space)
            self.prev_epoch = 0
        else:
            self._load_model()

        self.save_episode = save_episode

    def _save_model(self, epoch):
        self.agent.save_model(epoch)
        print("save model...")
        outfile = open('model/metadata.pkl', 'wb')
        pickle.dump(epoch, outfile)
        outfile.close()

    def _load_model(self):
        file = open('model/metadata.pkl', 'rb')
        self.prev_epoch = pickle.load(file)
        file.close()
        model = 'model/model_{}.h5'.format(self.prev_epoch)
        print("load model: {}".format(model))
        self.agent = DQN(self.action_space, self.state_space, init=False, model=model)

    def train(self, epoch):
        for e in range(1,epoch+1):
            state = self.env.reset()
            for _ in range(self.max_timestamp):
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                self.agent.replay()
                if done:
                    print("episode: {}/{}".format(e+self.prev_epoch, epoch+ self.prev_epoch))
                    break
            if e % self.save_episode == 0:
                self._save_model(e+self.prev_epoch)

if __name__ =="__main__":
    # initial training
    # paddleModel = PaddleModel(init=True)
    # paddleModel.train(300)

    # # continue train
    paddleModel = PaddleModel(init=False)
    paddleModel.train(200)