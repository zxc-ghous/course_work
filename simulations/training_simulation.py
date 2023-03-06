import os
import sys
import gymnasium as gym
import argparse
import sumo_rl

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

parser = argparse.ArgumentParser(description='simulations parameters')
parser.add_argument('--net_file',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\3_road_single_inter.net.xml')
parser.add_argument('--route_file',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\3_road_single_inter.rou.xml')
parser.add_argument('--out_csv_name',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\saved_history\history.csv')
args = parser.parse_args()


def create_env(use_gui=False, num_seconds=3600):
    env = gym.make('sumo-rl-v0',
                   net_file=args.net_file,
                   route_file=args.route_file,
                   out_csv_name=args.out_csv_name,
                   use_gui=use_gui,
                   num_seconds=num_seconds)

    return env


if __name__ == '__main__':
    env = create_env(True, 1)
    obs, _ = env.reset()
    print(env.action_space)
    print(env.observation_space)
    done = True
    while not done:
        next_obs, next_rew, terminated, truncated, info = env.step(env.action_space.sample())
