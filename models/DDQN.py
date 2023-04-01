import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from simulations.training_simulation import create_env
import argparse
import time
import os
import pprint

parser = argparse.ArgumentParser(description='DDQN parameters')
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.004)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.985)
parser.add_argument('--eps_min', type=float, default=0.01)
args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        self.model = self.create_model()

    def create_model(self):
        input_tensor = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(48, activation='relu',
                         kernel_initializer=tf.keras.initializers.RandomUniform(
                             minval=-0.3, maxval=0.3))(input_tensor)
        x = layers.Dense(24, activation='relu',
                         kernel_initializer=tf.keras.initializers.RandomUniform(
                             minval=-0.3, maxval=0.3))(x)
        output_tensor = layers.Dense(self.action_dim, activation='linear')(x)
        model = Model(input_tensor, output_tensor, name='DQN')
        model.compile(optimizer=Adam(learning_rate=args.lr), loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, save_name):
        self.model.save(os.path.join(r'C:\Users\sskil\PycharmProjects\course_work\saved_models', save_name))

    def load(self, folder_path):
        self.model = tf.keras.models.load_model(folder_path)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.reward_history = []

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states)[
                range(args.batch_size), np.argmax(self.model.predict(next_states), axis=1)]
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
            self.model.train(states, targets)

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            print(f'EPISODE {ep} started')
            ep_start_time = time.time()
            done, total_reward = False, 0
            state, _ = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.reward_history.append(reward)
                done = terminated or truncated
                self.buffer.put(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state

            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={} Time={}'.format(ep, total_reward, time.time() - ep_start_time))


if __name__ == '__main__':
    env = create_env(True, 5000)
    agent = Agent(env)
    agent.train(1)
    agent.model.save('3road_model')

