import config
from pymongo import MongoClient
import redis

import pandas as pd
from os import path
import csv


r = redis.StrictRedis(host='localhost', port=6379, db=0)

user = config.mongo["username"]
password = config.mongo["password"]
host = config.mongo["host"]
port = config.mongo["port"]
connection = MongoClient()

db = connection.gtfs_sptrans
all_code_lines_affected = list()
events_lng_lat = set()

max_distance_from_stop = 100
min_distance_from_stop = 0


def find_code_line_details(code_line):
    try:
        code_line_details = db.lines.find_one({"CodigoLinha": int(code_line)})
        return "{} / {}".format(code_line_details["DenominacaoTPTS"], code_line_details["DenominacaoTSTP"])
    except Exception as e:
        print(e)
        return ""


def find_code_line_by_direction(route_id, direction_id):
    if direction_id == 0:
        direction_id = 1
    else:
        direction_id = 2
    code_line_info = db.lines.find_one({"Letreiro": route_id, "Sentido": direction_id})
    if code_line_info is not None:
        return code_line_info["CodigoLinha"]
    return None


def find_trip_id_by_stop_id(stop_id):
    return db.stop_times.find_one({"stop_id": stop_id})


def find_affected_lines(latitude, longitude):
    code_lines_affected = set()
    # cache = False

    try:
        # results = r.get((longitude, latitude, max_distance_from_stop))
        # global cache
        # cache = results is not None
        # if cache:
        #     results = json.loads(results.decode('utf-8'))
        #     code_lines_affected.update(results)
        #     all_code_lines_affected.extend(results)
        #     return results
        # else:
        results = db.stops.find({"location": {"$near": {"$geometry": {"type": "Point", "coordinates": [longitude,
                                                                                                       latitude]},
                                                        "$minDistance": min_distance_from_stop,
                                                        "$maxDistance": max_distance_from_stop.fdiv(111.12)}}})
        for result in results:
            line_info = find_trip_id_by_stop_id(result["stop_id"])["trip_id"]
            line_details = line_info.split("-")
            route_id = line_details[0]
            if len(line_details) == 3:
                direction_id = line_details[2]
            else:
                direction_id = line_details[1]
            code_line = find_code_line_by_direction(route_id, direction_id)
            if code_line is not None:
                code_lines_affected.add(code_line)
    except Exception as e:
        print(e)

    # if not cache:
    #     r.set((longitude, latitude, max_distance_from_stop), list(code_lines_affected))
    all_code_lines_affected.extend(list(code_lines_affected))
    return list(code_lines_affected)


if __name__ == '__main__':
    df_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "notebooks",
                                      "processed_tweets.csv"), encoding='utf-8', sep=',', dtype={'_id': str})
    # df_events = df_events.loc[(df_events['label'] != "Irrelevante") & (df_events['address'].notnull()) &
    #                           (df_events["dateTime"].str.contains("20/02/2017"))]

    df_exception_events_with_address = \
        df_events.loc[(df_events['label'] != "Irrelevant") & (df_events['address'].notnull()) &
                      (df_events['lat'].notnull()) & (df_events['lng'].notnull())]

    df_exception_events = df_events.loc[(df_events['label'] != "Irrelevant")]

    print("Exception events: {}".format(len(df_exception_events)))
    print("Found address: {}".format(len(df_exception_events_with_address)))
    print("Found address percentage: {}".format(len(df_exception_events_with_address) / len(df_exception_events)))

    df_exception_events_with_address["affected_code_lines"] = \
        df_exception_events_with_address.apply(lambda x: find_affected_lines(x['lat'], x['lng']), axis=1)

    df_exception_events_with_address['dateTime'] = pd.to_datetime(df_exception_events_with_address.dateTime)
    df_exception_events_with_address.to_csv(
        path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                  "processed_tweets_affected_code_lines_{}.csv".format(max_distance_from_stop)), sep=",",
        index=False, quoting=csv.QUOTE_NONNUMERIC, header=True)

    df_code_lines = pd.DataFrame({"code_line": all_code_lines_affected})
    df_code_lines = df_code_lines.groupby('code_line')['code_line'].count()

    df_code_lines = df_code_lines.to_frame()
    df_code_lines['identification'] = df_code_lines.apply(lambda x: find_code_line_details(x.name), axis=1)

    df_code_lines.columns = ["qtd_exception_events", "identification"]
    df_code_lines.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "affected_lines.csv"),
                         sep=",", index=True, quoting=csv.QUOTE_NONNUMERIC, header=True)
