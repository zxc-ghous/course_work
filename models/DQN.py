import random
import numpy as np
from collections import deque
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from simulations.training_simulation import create_env


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        input_tensor = layers.Input(shape=(self.state_size,))
        x = layers.Dense(48, activation='relu')(input_tensor)
        x = layers.Dense(24, activation='relu')(x)
        output_tensor = layers.Dense(self.action_size, activation='linear')(x)
        model = Model(input_tensor, output_tensor, name='DQN')
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    history = []

    env = create_env(True,100)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(10):
        print(f'episode {e}')
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            print(reward)
            history.append(reward)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f'episode {e} is ended')
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    print(history[:300])
