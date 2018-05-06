import io
import zipfile

import pandas as pd
import pyproj as proj
from os import path
from shapely import geometry
from multiprocessing import Pool, cpu_count
import numpy as np
import csv
import calendar
import ast

# Set up projections
# WGS 84 (World Geodetic System) (aka WGS 1984, EPSG:4326)
crs_wgs = proj.Proj(init='epsg:4326')
# Use a locally appropriate projected CRS (Coordinate Reference System)
crs_bng = proj.Proj(init='epsg:29101')
bus_position_to_event_distance = 1000
events_lng_lat = set()
stats_frames = []


df_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                  "processed_tweets_CETSP_affected_code_lines_100.csv"), encoding='utf-8', sep=';')

df_events["dateTime"] = pd.to_datetime(df_events.dateTime)

df_exception_events_with_address = \
    df_events.loc[(df_events['_id'] == 833724664304250880)]

df_exception_events_with_address.apply(lambda x: events_lng_lat.add((x['lng'], x['lat'])), axis=1)

affected_code_lines = ast.literal_eval(df_exception_events_with_address['affected_code_lines'].values[0])
event_year = df_exception_events_with_address['dateTime'].tolist()[0].year
event_month = df_exception_events_with_address['dateTime'].tolist()[0].month
event_hour = df_exception_events_with_address['dateTime'].tolist()[0].hour

paths = []
weeks = calendar.monthcalendar(event_year, event_month)
for i in range(1, max(weeks[-1]) + 1):
    paths.append("_".join(["Movto", "{}{:02d}{:02d}{}00".format(event_year, event_month, i, event_hour - 1),
                           "{}{:02d}{:02d}{}00".format(event_year, event_month, i, event_hour)]))


cores = cpu_count()
partitions = cores


def parallelize(data, ref_function):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(ref_function, data_split))
    pool.close()
    pool.join()
    return data


def process_data(data):
    data["affected"] = data.apply(lambda x: is_close_to_event(x["nr_longitude_grau"], x["nr_latitude_grau"]), axis=1)
    return data


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


def func():
    for p in paths:
        try:
            archive = zipfile.ZipFile("/".join(['/Volumes', 'felipe', '{}.zip'.format(p)]), 'r')
            data = archive.read('{}.txt'.format(p))
            df = pd.read_csv(io.BytesIO(data), sep=';')
            df.columns = [
                "cd_evento_avl_movto", "cd_linha", "dt_movto", "nr_identificador", "nr_evento_linha", "nr_ponto",
                "nr_velocidade", "nr_voltagem", "nr_temperatura_interna", "nr_evento_terminal_dado", "nr_evento_es_1",
                "nr_latitude_grau", "nr_longitude_grau", "nr_indiceregistro", "dt_avl", "nr_distancia",
                "cd_tipo_veiculo_geo", "cd_avl_conexao", "cd_prefixo"
            ]

            df_affected_lines = df.loc[(df['cd_linha'].isin(affected_code_lines))]

            df_affected_lines = parallelize(df_affected_lines, process_data)

            df_affected_lines = df_affected_lines.loc[df_affected_lines["affected"]]

            df_stats = df_affected_lines.groupby('cd_linha')['nr_velocidade'].agg(['min', 'max', 'mean', 'var', 'std',
                                                                                   'count', pd.np.count_nonzero])
            df_stats["nonzero_percentage"] = (1 - df_stats['count_nonzero'] / df_stats['count']) * 100
            df_stats["filename"] = p.split("Movto_")[1]
            df_stats['dateTime'] = df_exception_events_with_address['dateTime'].tolist()[0]
            stats_frames.append(df_stats)
        except Exception as e:
            print(e)
            pass

    all_df = pd.concat(stats_frames)
    all_df.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "stats.csv"),
                  sep=";", index=True, quoting=csv.QUOTE_NONNUMERIC, header=True)


if __name__ == '__main__':
    func()
