import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from simulations.training_simulation import MySumoEnvironment
from simulations.training_simulation import args
from scipy import interpolate
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# TODO: system_total_stopped system_mean_waiting_time system_mean_speed running_cars
#       добавить kde графики

def plot_history(csv_file_path: str):
    data = pd.read_csv(csv_file_path)
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    sns.lineplot(data=data, x='step', y='system_total_stopped', ax=axes[0, 0], label='system_total_stopped')
    sns.lineplot(x=data['step'], y=[data['system_total_stopped'].median()] * len(data['step']),
                 linestyle='--', ax=axes[0, 0], label='median')
    sns.lineplot(data=data, x='step', y='system_running_cars', ax=axes[0, 1], label='system_running_cars')
    sns.lineplot(x=data['step'], y=[data['system_running_cars'].median()] * len(data['step']),
                 linestyle='--', ax=axes[0, 1], label='median')
    sns.lineplot(data=data, x='step', y='system_mean_waiting_time', ax=axes[1, 0], label='system_mean_waiting_time')
    sns.lineplot(x=data['step'], y=[data['system_mean_waiting_time'].median()] * len(data['step']),
                 linestyle='--', ax=axes[1, 0], label='median')
    sns.lineplot(data=data, x='step', y='system_mean_speed', ax=axes[1, 1], label='system_mean_speed')
    sns.lineplot(x=data['step'], y=[data['system_mean_speed'].median()] * len(data['step']),
                 linestyle='--', ax=axes[1, 1], label='median')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = MySumoEnvironment(args.net_file_2signle, args.route_file, single_agent=True, num_seconds=1000,
                            add_per_agent_info=False, out_csv_name=args.out_csv_name)
    obs, info = env.reset()
    for key, value in info.items():
        info.update({key: [value]})
    test_df = pd.DataFrame(info)
    done = False
    while not done:
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        for key, value in info.items():
            info.update({key: [value]})
        output = pd.DataFrame(info)
        test_df = pd.concat([test_df, output], ignore_index=True)
        done = terminated or truncated
