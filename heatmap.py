import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def build_heatmap():
    df = pd.read_csv("2016_2017.csv")
    df = df[df['score'] >= 3000]
    df = df[['day_of_week', 'hour']]

    df_agg = aggregate_posts_by_day(df)
    df_agg = df_agg.pivot(index='day', columns='hour', values='count')

    sns.heatmap(df_agg, linewidths = 0.2, square = True, fmt = 'g', yticklabels=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.axes().set_title("Heatmap of Reddit Submissions With >= 3000 Upvotes")
    plt.show()

def aggregate_posts_by_day(df):
    post_count = [[0 for x in range(24)] for y in range(7)]
    for index, row in df.iterrows():
        day, hour = row['day_of_week'], row['hour']
        post_count[day][hour] = post_count[day][hour] + 1

    post_count = [count for sublist in post_count for count in sublist] #flatten list
    #days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    days = [0,1,2,3,4,5,6]
    df_days = []
    for day in days:
        df_days = df_days + [day]*24
    df_hours = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]*7
    df_agg = pd.DataFrame({'day':df_days, 'hour':df_hours, 'count':post_count})
    return df_agg

if __name__ == "__main__":
    build_heatmap()
