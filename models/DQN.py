import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from simulations.training_simulation import create_env


# TODO: реализовать стратегию Больцмана заместо е-жадной стратегии
#       улучшение:прогнозная сеть
#       двойная DQN


# maxlen memory=10 000 до 1 000 000
# batch_size=между 32 и 2048
REPLAY_MEMORY_SIZE = 50000


class DQN:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()

    def create_model(self):
        input_tensor = layers.Input(shape=(self.state_size,))
        x = layers.Dense(48, activation='relu',
                         kernel_initializer=tf.keras.initializers.RandomUniform(
                             minval=-0.3, maxval=0.3))(input_tensor)
        x = layers.Dense(24, activation='relu',
                         kernel_initializer=tf.keras.initializers.RandomUniform(
                             minval=-0.3, maxval=0.3))(x)
        output_tensor = layers.Dense(self.action_size, activation='linear')(x)
        model = Model(input_tensor, output_tensor, name='DQN')
        model.compile(optimizer='adam', loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    env = create_env(True,360)
    state_size = 21
    action_size = 4
    agent = DQN(state_size, action_size)
    done = False
    batch_size = 32
    EPISODES = 1
    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        print(f'episode {e} started')
        for time in range(40):
            print(f'time {time} started')
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f'episode {e} done')
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
