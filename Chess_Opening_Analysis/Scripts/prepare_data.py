# Data cleaning V1.0
# 02/12/2021

import sys
import pandas
import csv

def main():
    if len(sys.argv) < 3:
        sys.exit(f'Missing parameter(s), try: "Python   {sys.argv[0]} original_csv, columns_new_csv, path_new_csv"')

    with open(sys.argv[2]) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        columns = list(csv_reader)[0]

    with open(sys.argv[1]) as csv_file:
        df = pandas.read_csv(csv_file)
        df = df[columns]

    new_csv = df.to_csv({sys.argv[3]})

def get_base_time(time_control):
    time_control = time_control.split('+')
    return time_control[0]

def csv_transform(filename):
    with open(filename) as csv_file:
        df = pandas.read_csv(csv_file)
        df = df[['White Elo', 'Black Elo', 'TimeControl', 'Result', 'Moves']]
        for index in range(len(df['Moves'])):
            game = df['Moves'][index]
            new_string = ''
            raw_data = game.split('.')

            for i in range(1, len(raw_data)):
                new_string += raw_data[i][0:-2]
            df.at[index, 'Moves'] = new_string[1:]

        for index in range(len(df['TimeControl'])):
            time_control = df['TimeControl'][index]
            base_time = get_base_time(time_control)
            df.at[index, 'TimeControl'] = base_time

    new_csv = df.to_csv('data/cleaned data/2016_CvC_cleaned.csv')


csv_transform('Data/Games/2016_CvC.csv')
