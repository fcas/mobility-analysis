from multiprocessing.pool import Pool

import pandas as pd
import config
import matplotlib.pyplot as plt
from os import path
from pymongo import MongoClient
from shapely import geometry
import pyproj as proj
import redis
import numpy as np
from multiprocessing import cpu_count
import dask.dataframe as dd
import csv
import json


r = redis.StrictRedis(host='localhost', port=6379, db=0)

user = config.mongo["username"]
password = config.mongo["password"]
host = config.mongo["host"]
port = config.mongo["port"]
connection = MongoClient()

db = connection.gtfs_sptrans
all_code_lines_affected = list()
events_lng_lat = set()

stats_frames = []

# Set up projections
# WGS 84 (World Geodetic System) (aka WGS 1984, EPSG:4326)
crs_wgs = proj.Proj(init='epsg:4326')
# Use a locally appropriate projected CRS (Coordinate Reference System)
crs_bng = proj.Proj(init='epsg:29101')

bus_position_to_event_distance = 1000
max_distance_from_shape = 100
min_distance_from_shape = 0


# Casting geographic coordinate pair to the projected system
def project_lng_lat(input_lng, input_lat):
    return proj.transform(crs_wgs, crs_bng, input_lng, input_lat)


def is_close_to_event(lng_bus_position, lat_bus_position):
    point_2 = geometry.Point(project_lng_lat(lng_bus_position, lat_bus_position))
    for coordinate in events_lng_lat:
        point_1 = geometry.Point(project_lng_lat(coordinate[0], coordinate[1]))
        if point_1.distance(point_2) < bus_position_to_event_distance:
            return True
    return False


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


def find_trip_id_by_shape_id(shape_id):
    return db.trips.find_one({"shape_id": shape_id})


def find_affected_lines(latitude, longitude):
    events_lng_lat.add((longitude, latitude))
    code_lines_affected = set()
    cache = False

    try:
        results = r.get((longitude, latitude, max_distance_from_shape))
        global cache
        cache = results is not None
        if cache:
            results = json.loads(results.decode('utf-8'))
            code_lines_affected.update(results)
            all_code_lines_affected.extend(results)
            return results
        else:
            results = db.shapes.find({"location": {"$near": {"$geometry": {"type": "Point",
                                      "coordinates": [longitude, latitude]}, "$minDistance": min_distance_from_shape,
                                                             "$maxDistance": max_distance_from_shape}}})
            for result in results:
                shape_id = result["shape_id"]
                line_info = find_trip_id_by_shape_id(shape_id)["trip_id"]
                line_details = line_info.split("-")
                route_id = line_details[0]
                if len(line_details) == 3:
                    direction_id = line_details[2]
                else:
                    direction_id = line_details[1]
                code_line = find_code_line_by_direction(route_id, direction_id)
                if code_line is not None:
                    code_lines_affected.add(code_line)
                    all_code_lines_affected.append(code_line)
    except Exception as e:
        print(e)

    if not cache:
        r.set((longitude, latitude, max_distance_from_shape), list(code_lines_affected))
    return list(code_lines_affected)


def get_close_events(data_frame):
    df_affected_lines["affected"] = data_frame.apply(lambda x: is_close_to_event(x["nr_longitude_grau"],
                                                                                 x["nr_latitude_grau"]), axis=1)
    return df_affected_lines


if __name__ == '__main__':
    df_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                      "processed_tweets_CETSP_.csv"), encoding='utf-8', sep=';')
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

    df_exception_events_with_address.to_csv(
        path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                  "processed_tweets_CETSP_affected_code_lines_{}.csv".format(max_distance_from_shape)), sep=";",
        index=False, quoting=csv.QUOTE_NONNUMERIC, header=True)

#
#
# paths = ["Movto_201702192300_201702200000", "Movto_201702200000_201702200100", "Movto_201702200100_201702200200",
#          "Movto_201702200200_201702200300", "Movto_201702200300_201702200400", "Movto_201702200400_201702200500",
#          "Movto_201702200500_201702200600", "Movto_201702200600_201702200700", "Movto_201702200700_201702200800",
#          "Movto_201702200800_201702200900", "Movto_201702200900_201702201000", "Movto_201702201000_201702201100",
#          "Movto_201702201100_201702201200", "Movto_201702201200_201702201300", "Movto_201702201300_201702201400",
#          "Movto_201702201400_201702201500", "Movto_201702201500_201702201600", "Movto_201702201600_201702201700",
#          "Movto_201702201700_201702201800", "Movto_201702201800_201702201900", "Movto_201702201900_201702202000",
#          "Movto_201702202000_201702202100", "Movto_201702202100_201702202200", "Movto_201702202200_201702202300"]

# paths = ["Movto_201702201300_201702201400"]

    df_code_lines = pd.DataFrame({"code_line": all_code_lines_affected})
    df_code_lines = df_code_lines.groupby('code_line')['code_line'].count()

    df_code_lines = df_code_lines.to_frame()
    df_code_lines['identification'] = df_code_lines.apply(lambda x: find_code_line_details(x.name), axis=1)

    df_code_lines.columns = ["qtd_exception_events", "identification"]
    df_code_lines.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "affected_lines.csv"),
                         sep=";", index=True, quoting=csv.QUOTE_NONNUMERIC, header=True)

    # for p in paths:
    #     df = dd.read_csv("/Volumes/felipetoshiba/SPTrans/Fevereiro/" + p + ".txt",
    #                      encoding='latin-1', sep=';')
    #     df.columns = [
    #         "cd_evento_avl_movto", "cd_linha", "dt_movto", "nr_identificador", "nr_evento_linha", "nr_ponto",
    #         "nr_velocidade", "nr_voltagem", "nr_temperatura_interna", "nr_evento_terminal_dado", "nr_evento_es_1",
    #         "nr_latitude_grau", "nr_longitude_grau", "nr_indiceregistro", "dt_avl", "nr_distancia",
    #         "cd_tipo_veiculo_geo", "cd_avl_conexao", "cd_prefixo"
    #     ]
    #
    #     df_affected_lines = df.loc[(df['cd_linha'].isin(set(all_code_lines_affected)))]
    #     df_affected_lines["affected"] = df.apply(lambda x: is_close_to_event(x["nr_longitude_grau"],
    #                                                                          x["nr_latitude_grau"]), axis=1)
    #
    #     df_affected_lines = df_affected_lines.loc[df_affected_lines["affected"]]
    #     df_affected_lines = df_affected_lines.compute()
    #
    #     df_stats = df_affected_lines.groupby('cd_linha')['nr_velocidade'].agg(['min', 'max', 'mean', 'var', 'std',
    #                                                                            'count', pd.np.count_nonzero])
    #     df_stats["nonzero_percentage"] = (1 - df_stats['count_nonzero'] / df_stats['count']) * 100
    #     stats_frames.append(df_stats)

    # # stats("/Volumes/felipetoshiba/SPTrans/Fevereiro/Movto_201702192300_201702200000.txt")
    # all_df = pd.concat(stats_frames)
    # all_df.hist(column='mean')
    # plt.show()