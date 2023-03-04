import os
import sys
import gymnasium as gym
import sumo_rl
import traci
from tf_agents.environments import tf_py_environment, gym_wrapper, suite_gym

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def create_env(use_gui=False, num_seconds=3600):
    env = gym.make('sumo-rl-v0',
                   net_file=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\2way_single_intersection.net.xml',
                   route_file=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\2way_single_intersection.rou.xml',
                   out_csv_name=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\history.csv',
                   use_gui=use_gui,
                   num_seconds=num_seconds)

    return env


if __name__ == '__main__':
    env = create_env(True,1)
    obs, _ = env.reset()
    done = False
    while not done:
        next_obs, next_rew, terminated, truncated, info = env.step(env.action_space.sample())
