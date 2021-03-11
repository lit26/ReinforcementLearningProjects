from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='relu'))
    return model