import os
import sys
import gymnasium as gym
import argparse
from sumo_rl import SumoEnvironment
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

parser = argparse.ArgumentParser(description='simulations parameters')
parser.add_argument('--net_file_single',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk_single_lights.net.xml')
parser.add_argument('--net_file_2single',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk_2single_lights.net.xml')
parser.add_argument('--route_file',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk.rou.xml')
parser.add_argument('--out_csv_name',
                    type=str,
                    default=r'C:\Users\sskil\PycharmProjects\course_work\saved_history\history.csv')
args = parser.parse_args()


# deprecated
def create_single_env(use_gui=False, num_seconds=3500, out_csv_name=args.out_csv_name):
    env = MySumoEnvironment(net_file=args.net_file_single, route_file=args.route_file,
                            out_csv_name=out_csv_name, use_gui=use_gui,
                            num_seconds=num_seconds, add_per_agent_info=False, single_agent=True)

    return env


# deprecated
def create_2single_env(use_gui=False, num_seconds=3500, out_csv_name=args.out_csv_name):
    env = MySumoEnvironment(net_file=args.net_file_2signle, route_file=args.route_file,
                            out_csv_name=out_csv_name, use_gui=use_gui,
                            num_seconds=num_seconds, add_per_agent_info=False, single_agent=True)

    return env


def create_env(net_file, route_file,
               out_csv_name=args.out_csv_name, use_gui=False, num_seconds=3600):
    env = MySumoEnvironment(net_file=net_file, route_file=route_file,
                            out_csv_name=out_csv_name, use_gui=use_gui,
                            num_seconds=num_seconds, add_per_agent_info=False,
                            single_agent=True, sumo_warnings=False)
    return env


class MySumoEnvironment(SumoEnvironment):
    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            # add system_running_cars to info dictionary
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "system_running_cars": len(vehicles)
        }


if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env

    env = make_vec_env(create_env, env_kwargs=dict(net_file=args.net_file_2single, route_file=args.route_file,
                                                   use_gui=True))
    env.reset()
    while True:
        obs, rewards, dones, infos = env.step(np.array([env.action_space.sample()]))
        print(infos)
