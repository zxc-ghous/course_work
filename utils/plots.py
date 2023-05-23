import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.parse_summary import parse_summary
from simulations.training_simulation import MySumoEnvironment
from simulations.training_simulation import args
from scipy import interpolate
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# TODO: system_total_stopped system_mean_waiting_time system_mean_speed running_cars
#       добавить kde графики

def plot_history(data=None, csv_file_path=None):
    if csv_file_path:
        data = pd.read_csv(csv_file_path)
    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    sns.lineplot(data=data, x='step', y='system_total_stopped', ax=axes[0, 0], label='system_total_stopped')
    sns.lineplot(x=data['step'], y=[data['system_total_stopped'].median()] * len(data['step']),
                 linestyle='--', ax=axes[0, 0], label=f'median={data["system_total_stopped"].median()}')
    sns.lineplot(data=data, x='step', y='system_running_cars', ax=axes[0, 1], label='system_running_cars')
    sns.lineplot(x=data['step'], y=[data['system_running_cars'].median()] * len(data['step']),
                 linestyle='--', ax=axes[0, 1], label=f'median={data["system_running_cars"].median()}')
    sns.lineplot(data=data, x='step', y='system_mean_waiting_time', ax=axes[1, 0], label='system_mean_waiting_time')
    sns.lineplot(x=data['step'], y=[data['system_mean_waiting_time'].median()] * len(data['step']),
                 linestyle='--', ax=axes[1, 0], label=f'median={data["system_mean_waiting_time"].median()}')
    sns.lineplot(data=data, x='step', y='system_mean_speed', ax=axes[1, 1], label='system_mean_speed')
    sns.lineplot(x=data['step'], y=[data['system_mean_speed'].median()] * len(data['step']),
                 linestyle='--', ax=axes[1, 1], label=f'median={data["system_mean_speed"].median()}')
    fig.tight_layout()
    plt.show()


def compare_history(data_irl, data_net):
    data_irl = data_irl[
        ['system_running_cars', 'system_total_stopped', 'system_mean_waiting_time', 'system_mean_speed']]
    data_irl.insert(0, 'class', 'irl')
    data_net = data_net[
        ['system_running_cars', 'system_total_stopped', 'system_mean_waiting_time', 'system_mean_speed']]
    data_net.insert(0, 'class', 'net')

    data = pd.concat([data_irl, data_net], axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    for col, ax in zip(data.columns[1:], axes.ravel()):
        sns.boxplot(data=data, x='class', y=col, ax=ax, width=0.4)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_net = pd.read_csv(r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_A2C\history_conn0_run27.csv")
    summary = parse_summary(r'C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk_summary.xml')
    compare_history(summary, data_net)

    #plot_history(csv_file_path=r"C:\Users\sskil\PycharmProjects\course_work\saved_history\sb3_A2C\history_conn0_run27.csv")
    #plot_history(data=summary)
