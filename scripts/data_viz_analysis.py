import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import path
import csv


def plot_affected_lines():
    df_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "affected_lines.csv"),
                            encoding='utf-8', sep=';')

    plt.figure(figsize=(24, 16))
    plt.scatter(x=df_events['code_line'], y=df_events['qtd_exception_events'], alpha=0.6)
    plt.xlabel('Quantity of exception events', fontsize=22)
    plt.ylabel('Bus line codes', fontsize=22)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.grid(True)
    plt.show()

    pt = pd.pivot_table(df_events, values=["qtd_exception_events"], index=['code_line', 'identification'])
    pt.sort_values(by='qtd_exception_events', ascending=False, inplace=True)
    print(pt)


def plot_affected_addresses():
    processed_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                             "processed_tweets_CETSP_.csv"), encoding='utf-8', sep=';')
    processed_events = processed_events.loc[(processed_events['label'] != "Irrelevant") &
                                            (processed_events['address'].notnull()) &
                                            (processed_events['lat'].notnull()) & (processed_events['lng'].notnull())]

    processed_events["address"] = processed_events.apply(lambda x: x["address"].split(",")[0].split("-")[0], axis=1)
    processed_events = processed_events.groupby('address')['address'].count()

    processed_events = processed_events.to_frame()
    processed_events = processed_events.sort_values(by=['address'], ascending=False)

    processed_events.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                      "affected_addresses.csv"), sep=";", index=True, quoting=csv.QUOTE_NONNUMERIC,
                            header=True)

    index = list(range(0, len(processed_events.address[0:50])))
    plt.figure(figsize=(24, 16))
    plt.bar(index, processed_events.address[0:50], align='center')
    plt.ylabel('Quantity of exception events', fontsize=22)
    plt.xlabel('Address', fontsize=22)
    plt.xticks(index, processed_events.index.values[0:50], rotation='vertical', fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.show()


def plot_stats():
    df_stats = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                     "stats_894696768767762432.csv"), encoding='utf-8', sep=';')
    df_stats['day'] = df_stats.apply(lambda x: int(x['filename'].split("_")[0][6:8]), axis=1)
    event_datetime = datetime.datetime.strptime(df_stats['dateTime'].values[0], '%Y-%m-%d %H:%M:%S')
    code_lines = list(set(df_stats['cd_linha'].tolist()))

    for code_line in code_lines:
        df = df_stats.loc[(df_stats['cd_linha'] == code_line)]
        filename = df['filename'].tolist()[0]
        year = filename[:4]
        month = filename[4:6]
        from_hour = filename[8:10]
        to_hour = filename[21:23]
        month_name = datetime.date(int(year), int(month), 1).strftime('%B')

        index = list(range(0, len(df)))
        plt.figure(figsize=(24, 16))
        plt.bar(index, df['mean'], align='center', color=list(
            np.where(df["day"] == event_datetime.day, '#FF6851', '#647AFF')))
        plt.title("Velocity means to code line {}".format(code_line), fontsize=30)
        plt.ylabel('Velocity mean', fontsize=22)
        plt.xlabel('{} days between {} and {} hours, {}. Event time {}'
                   .format(month_name, from_hour, to_hour, year, event_datetime), fontsize=22)
        plt.xticks(index, df['day'].values, fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)

        # fig = plt.figure(figsize=(24, 16))
        # ax = fig.add_subplot(111)
        # ax2 = ax.twinx()
        #
        # width = 0.4
        #
        # df.loc[:, ['mean', 'median' 'max', 'std']].plot.bar(ax=ax, width=width, position=1)
        # df.loc[:, ['nonzero_percentage']].plot.bar(ax=ax2, width=width, position=0, color="gray")
        #
        # ax.set_xlabel('{} days between {} and {} hours, {}. Event time {}'
        #               .format(month_name, from_hour, to_hour, year, event_datetime), fontsize=22)
        # plt.xticks(index, df['day'].values, fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.grid(True)
        # ax.set_ylabel('Velocity', fontsize=20)
        # ax2.set_ylabel('Zero percentage', fontsize=20)
        #
        # for tick in ax.xaxis.get_ticklabels():
        #     tick.set_fontsize(20)
        #     tick.set_rotation('horizontal')
        #
        # for tick in ax.yaxis.get_ticklabels():
        #     tick.set_fontsize(20)
        #
        # plt.legend(loc="upper left")

        plt.show()


df = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                           "processed_tweets_CETSP_affected_code_lines_100.csv"), encoding='utf-8', sep=';')
df["dateTime"] = pd.to_datetime(df.dateTime)
df["lat"] = df['lat'].astype(str)
df["lng"] = df['lng'].astype(str)
df.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                    "2_processed_tweets_CETSP_affected_code_lines_100.csv"), sep=",", index=False,
          quoting=csv.QUOTE_NONNUMERIC, header=True)
