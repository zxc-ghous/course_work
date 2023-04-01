import tensorflow as tf
from keras.layers import Input, Dense
import time
import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
import gymnasium as gym
import sumo_rl
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use cpu

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--net_file',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\3_road_single_inter.net.xml')
parser.add_argument('--route_file',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\3_road_poission_slow.rou.xml')
parser.add_argument('--out_csv_name',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\saved_history\A3C_history\history.csv')
args = parser.parse_args()

CUR_EPISODE = 0


def create_env(use_gui=True, num_seconds=5000):
    env = gym.make('sumo-rl-v0',
                   net_file=args.net_file,
                   route_file=args.route_file,
                   out_csv_name=args.out_csv_name,
                   use_gui=use_gui,
                   num_seconds=num_seconds,
                   add_per_agent_info=False)

    return env


class Actor:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, create_env_fn):
        self.create_env = create_env_fn
        env = self.create_env()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.global_actor = Actor(self.state_dim, self.action_dim)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = 4  # cpu_count() = 8

    def train(self, max_episodes=1000):
        workers = []
        for i in range(self.num_workers):
            env = self.create_env()
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        Thread.__init__(self)
        self.lock = Lock()
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    @staticmethod
    def n_step_td_target(rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    @staticmethod
    def advantage(td_targets, baselines):
        return td_targets - baselines

    @staticmethod
    def list_to_batch(_list):
        batch = _list[0]
        for elem in _list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        global CUR_EPISODE

        while self.max_episodes > CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state, _ = self.env.reset()

            while not done:
                probs = self.actor.model.predict(np.reshape(state, [1, self.state_dim]), verbose=0)
                action = np.random.choice(self.action_dim, p=probs[0])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state, verbose=0)
                    td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states, verbose=0)

                    with self.lock:
                        actor_loss = self.global_actor.train(states, actions, advantages)
                        critic_loss = self.global_critic.train(states, td_targets)

                        self.actor.model.set_weights(self.global_actor.model.get_weights())
                        self.critic.model.set_weights(self.global_critic.model.get_weights())

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            CUR_EPISODE += 1

    def run(self):
        self.train()


if __name__ == '__main__':
    agent = Agent(create_env)
    start = time.time()
    agent.train(8)
    print(f'lr time {time.time() - start}')
