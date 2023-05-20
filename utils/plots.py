import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def plot_history(csv_file_path: str, system_info=True):
    data = pd.read_csv(csv_file_path)
    if system_info:
        fig, axes = plt.subplots(2, 2, figsize=(15, 6))
        sns.lineplot(data=data, x='step', y='system_total_stopped', ax=axes[0, 0])
        sns.lineplot(data=data, x='step', y='system_total_waiting_time', ax=axes[0, 1])
        sns.lineplot(data=data, x='step', y='system_mean_waiting_time', ax=axes[1, 0])
        sns.lineplot(data=data, x='step', y='system_mean_speed', ax=axes[1, 1])
        sns.lineplot(data=data, x='step', y=data['system_mean_speed'].mean(), ax=axes[1, 1])
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_history(r'C:\Users\sskil\PycharmProjects\course_work\saved_history\A3C_history\history.csv_conn4_run3.csv', True)
