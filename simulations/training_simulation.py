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
parser.add_argument('--net_file_single',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk_single_lights.net.xml')
parser.add_argument('--net_file_2signle',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk_2single_lights.net.xml')
parser.add_argument('--route_file',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk.rou.xml')
parser.add_argument('--out_csv_name',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\saved_history\history.csv')
args = parser.parse_args()


def create_single_env(use_gui=False, num_seconds=3500, out_csv_name=args.out_csv_name):
    env = gym.make('sumo-rl-v0',
                   net_file=args.net_file_single,
                   route_file=args.route_file,
                   out_csv_name=out_csv_name,
                   use_gui=use_gui,
                   num_seconds=num_seconds,
                   add_per_agent_info=False)

    return env


def create_2single_env(use_gui=False, num_seconds=3500, out_csv_name=args.out_csv_name):
    env = gym.make('sumo-rl-v0',
                   net_file=args.net_file_2signle,
                   route_file=args.route_file,
                   out_csv_name=r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_PPO\history",
                   use_gui=use_gui,
                   num_seconds=num_seconds,
                   add_per_agent_info=False)

    return env


if __name__ == '__main__':
    pass