import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def plot_history(csv_file_path: str, system_info=True, agent_info=True):
    data = pd.read_csv(csv_file_path)
    if system_info:
        fig, axes = plt.subplots(2, 2, figsize=(15, 6))
        sns.lineplot(data=data, x='step', y='system_total_stopped', ax=axes[0, 0])
        sns.lineplot(data=data, x='step', y='system_total_waiting_time', ax=axes[0, 1])
        sns.lineplot(data=data, x='step', y='system_mean_waiting_time', ax=axes[1, 0])
        sns.lineplot(data=data, x='step', y='system_mean_speed', ax=axes[1, 1])
        fig.tight_layout()
        plt.show()
    if agent_info:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.lineplot(data=data, x='step', y='agents_total_stopped', ax=axes[0])
        sns.lineplot(data=data, x='step', y='agents_total_accumulated_waiting_time', ax=axes[1])
        fig.tight_layout()
        plt.show()

