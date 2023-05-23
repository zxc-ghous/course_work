import pandas as pd
import csv
import re


def parse_summary(summary_path):
    data = []
    with open(summary_path, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:
            if (row):
                data.append(row)

    data = data[13:-1]
    data_fr = []
    for i in data:
        result = re.findall(r'[0-9]*[.,]?[0-9]+', i[0])
        data_fr.append(list(map(float, result)))

    columns = ['step', 'loaded', 'inserted', 'system_running_cars', 'waiting', 'ended', 'arrived', 'collisions',
               'teleports', 'system_total_stopped', 'stopped', 'system_mean_waiting_time',
               'meanTravelTime', 'system_mean_speed', 'meanSpeedRelative', 'duration']
    data_fr = pd.DataFrame(data_fr, columns=columns)
    return data_fr


if __name__ == "__main__":
    summary_path = r"C:\Users\sskil\PycharmProjects\course_work\sumo_env\tomsk\tomsk_summary.xml"
    ttt = parse_summary(summary_path)
    print(ttt)
