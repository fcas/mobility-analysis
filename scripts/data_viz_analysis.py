import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import path
import csv
import glob

from pandas.errors import EmptyDataError


def remove_last_digits(address):
    last_string = address.split(" ")[-1]
    digits = filter(lambda x: x.isdigit(), last_string)
    for digit in digits:
        address = address.replace(digit, "")
    return address


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
    processed_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "notebooks",
                                             "processed_tweets.csv"), encoding='utf-8', sep=',')
    processed_events = processed_events.loc[(processed_events['label'] != "Irrelevant") &
                                            (processed_events['address'].notnull()) &
                                            (processed_events['location_type'] != "APPROXIMATE") &
                                            (processed_events['lat'].notnull()) & (processed_events['lng'].notnull())]

    processed_events["address"] = processed_events.apply(lambda x: x["address"]
                                                         .replace(", São Paulo - SP, Brazil", "")
                                                         .replace(", São Paulo - SP, ", "")
                                                         .replace(", Brazil", "")
                                                         .replace("Rua", "R.")
                                                         .replace("Avenida", "Av.")
                                                         .replace("Ponte", "Pte.")
                                                         .replace("Brigadeiro", "Brg. ")
                                                         .replace("Deputado", "Dep. ")
                                                         .replace("Engenheiro", "Eng. ")
                                                         .replace("General", "Gen. ")
                                                         .replace("Capitão", "Cap. ")
                                                         .replace("Coronel", "Cel. ")
                                                         .replace("Professor", "Prof. ")
                                                         .replace("Professora", "Profa. ")
                                                         .replace("Visconde", "Visc. ")
                                                         .replace("Senador", "Sen. ")
                                                         .replace("Praça", "Pça. ")
                                                         .replace("Travessa", "Tv.").split(",")[0].split("-")[0]
                                                         .strip(), axis=1)

    processed_events["address"] = processed_events.apply(lambda x: remove_last_digits(x["address"]),
                                                         axis=1)

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
                                     "stats_867839850661072896.csv"), encoding='utf-8', sep=',')
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


def describe(month, event_type):
    distance = 1000
    csv_file = open("velocity_analysis_{}_{}m_{}.csv".format(month, distance, event_type), "w")
    writer = csv.writer(csv_file, delimiter=',')

    for file_path in glob.glob(path.join(
            path.dirname(path.realpath(__file__)), "..", "datasets",
            "data_2017_{:02}_exception_events_{}m", "stats_*_{}.csv").format(month, distance, event_type)):

        file_name = file_path.split("datasets/")[1]
        event_id = file_name.split("stats_")[1].split("_")[0]

        try:
            df_stats = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                             file_name), encoding='utf-8', sep=',')
            if not df_stats.empty:
                df_stats['day'] = df_stats.apply(lambda x: int(x['filename'].split("_")[0][6:8]), axis=1)
                try:
                    event_datetime = datetime.datetime.strptime(df_stats['dateTime'].values[0], '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    event_datetime = datetime.datetime.strptime(df_stats['dateTime'].values[0], '%Y-%m-%d')
                    print(e)
                    pass
                code_lines = list(set(df_stats['cd_linha'].tolist()))
                velocity_mean = np.mean(df_stats['median'].tolist())
                for code_line in code_lines:
                    df_event_day = df_stats.loc[(df_stats['cd_linha'] == code_line) & (df_stats['day'] ==
                                                                                       event_datetime.day)]
                    if len(list(df_event_day['mean'])) == 0:
                        pass
                    else:
                        writer.writerow([event_id, code_line, str(event_datetime),
                                         1 if float(df_event_day['median']) <= velocity_mean else 0])
            else:
                pass
        except EmptyDataError:
            pass

    csv_file.close()

    try:
        velocities = pd.read_csv('velocity_analysis_{}_{}m_{}.csv'.format(month, distance, event_type),
                                 encoding='utf-8', sep=',', header=None)
        velocities.columns = ['event_id', 'code_line', 'dateTime', 'less_count']

        grouped_ids = velocities.groupby(['event_id'])

        up = 0
        down = 0
        for name, group in grouped_ids:
            less = len(group.loc[(group['less_count'] == 1)])
            more = len(group.loc[(group['less_count'] == 0)])
            if less / (more + less) > 0.5:
                down = down + 1
            else:
                up = up + 1

        if down + up > 0:
            print("{}:{}".format(event_type, (down / (down + up)) * 100))
        else:
            print("{}:{}".format(event_type, 0))
    except Exception as e:
        print(e)
        pass

# df = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
#                            "processed_tweets_CETSP_affected_code_lines_100.csv"), encoding='utf-8', sep=';')
# df["dateTime"] = pd.to_datetime(df.dateTime)
# df["lat"] = df['lat'].astype(str)
# df["lng"] = df['lng'].astype(str)
# df.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
#                     "2_processed_tweets_CETSP_affected_code_lines_100.csv"), sep=",", index=False,
#           quoting=csv.QUOTE_NONNUMERIC, header=True)

# plot_affected_addresses()
# plot_stats()


if __name__ == '__main__':
    months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
              "september": 9, "october": 10, "november": 11, "december": 12}
    events = ["accident", "natural_disaster", "social_event", "urban_event"]

    for key in months.keys():
        print(key)
        for event in events:
            describe(months[key], event)
