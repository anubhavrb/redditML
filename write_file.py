import numpy as np
import pandas as pd
import datetime as dt

"""
Function that reads in the csv files for the 2016 and 2017 data.
"""
def parse_csv():
    df_2016 = pd.read_csv("2016.csv")
    df_2017 = pd.read_csv("2017.csv")
    df = pd.concat([df_2016,df_2017])
    return df

"""
Function that writes the concatenated df to a combined csv file.
"""
def write_combined_csv(df):
    df = df.fillna("")
    df.to_csv("2016_2017.csv")

"""
Function that adds columns for day of year, day of week, hour of day, and
minute of hour.
"""
def add_time_columns(row):
    utc_time = row['created_utc']
    day_of_year = dt.datetime.fromtimestamp(int(utc_time)).strftime('%-j')
    day_of_week = dt.datetime.fromtimestamp(int(utc_time)).strftime('%w')
    hour = dt.datetime.fromtimestamp(int(utc_time)).strftime('%-H')
    minute = dt.datetime.fromtimestamp(int(utc_time)).strftime('%-M')
    return pd.Series([day_of_year, day_of_week, hour, minute], index = ['day_of_year', 'day_of_week', 'hour', 'minute'])


"""
Function that adds a target column to the dataset, on the basis of the score:
Score   Bin
0-1     0 (65th percentile)
2-7     1 (95th percentile)
8-43    2 (99th percentile)
44-end  3 (100th percentile)
"""
def add_popularity_column(row):
    score = row['score']
    popularity = 0
    if score < 2:
        popularity = 0
    elif score < 8:
        popularity = 1
    elif score < 44:
        popularity = 2
    else:
        popularity = 4
    return popularity

"""
Function that reads in the 2016 and 2017 data, combines them, adds columns for
specific times, and writes the combined csv file.
"""
def main():
    df = parse_csv()
    df[['day_of_year', 'day_of_week', 'hour', 'minute']] = df.apply(lambda row: add_time_columns(row), axis=1)
    write_combined_csv(df)

def change_columns(row):
    var1 = row['quality'] + 5
    var2 = row['quality'] + 10
    var3 = row['quality'] + 15
    return pd.Series([var1, var2, var3])

if __name__ == "__main__":
    df = pd.read_csv("2016_2017.csv")
    df['popularity'] = df.apply(lambda row: add_popularity_column(row), axis=1)
    df.to_csv("2016_2017.csv")
