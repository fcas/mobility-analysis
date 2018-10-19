import io
from threading import Thread
from queue import Queue
import zipfile

import pandas as pd
import pyproj as proj
from os import path
import glob
from shapely import geometry
from multiprocessing import Pool, cpu_count
import numpy as np
import csv
import ast

# Set up projections
# WGS 84 (World Geodetic System) (aka WGS 1984, EPSG:4326)
from mov_to_generator import get_file_paths

crs_wgs = proj.Proj(init='epsg:4326')
# Use a locally appropriate projected CRS (Coordinate Reference System)
crs_bng = proj.Proj(init='epsg:29101')
bus_position_to_event_distance = 1000
events_lng_lat = set()

cores = cpu_count()
partitions = cores
processed = []
for file in glob.glob(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "stats_*")):
    processed.append(file.split("_")[1].split(".csv")[0])

processed = list(set(processed))

num_fetch_threads = 1
enclosure_queue = Queue()


def process_movto_files(paths, event_id, event_date_time, event_affected_code_lines, event_lng, event_lat):
    stats_frames = []
    print("Processing event id: {} ".format(event_id))
    print("Paths: {} ".format(paths))
    for p in paths:
        print("Processing {} file".format(p))
        archive = None
        try:
            archive = zipfile.ZipFile("/".join(['/Volumes', 'felipe', '{}.zip'.format(p)]), 'r')
        except Exception as e:
            print(e)
            pass

        if archive is not None:
            data = archive.read('{}.txt'.format(p))
            df = pd.read_csv(io.BytesIO(data), sep=';')
            df.columns = [
                "cd_evento_avl_movto", "cd_linha", "dt_movto", "nr_identificador", "nr_evento_linha", "nr_ponto",
                "nr_velocidade", "nr_voltagem", "nr_temperatura_interna", "nr_evento_terminal_dado", "nr_evento_es_1",
                "nr_latitude_grau", "nr_longitude_grau", "nr_indiceregistro", "dt_avl", "nr_distancia",
                "cd_tipo_veiculo_geo", "cd_avl_conexao", "cd_prefixo"
            ]

            global events_lng_lat
            events_lng_lat.clear()
            events_lng_lat.add((float(event_lng), float(event_lat)))

            df_affected_lines = df.loc[(df['cd_linha'].isin(event_affected_code_lines))]
            if not df_affected_lines.empty:
                df_affected_lines = parallelize(df_affected_lines, process_data)
                df_affected_lines = df_affected_lines.loc[df_affected_lines["affected"]]

                df_stats = df_affected_lines.groupby('cd_linha')['nr_velocidade'].agg(['min', 'max', 'mean', 'median',
                                                                                       'var', 'std', 'count',
                                                                                       pd.np.count_nonzero])
                df_stats["nonzero_percentage"] = (1 - df_stats['count_nonzero'] / df_stats['count']) * 100
                df_stats["filename"] = p.split("Movto_")[1]
                df_stats['dateTime'] = event_date_time
                stats_frames.append(df_stats)
            else:
                print("No records related to affected lines in {} file".format(p))

    if len(stats_frames) > 0:
        all_df = pd.concat(stats_frames)
        all_df.to_csv(path.join(path.dirname(path.realpath('__file__')),
                                "..", "datasets", "stats_{}.csv".format(event_id)), sep=",", index=True,
                      quoting=csv.QUOTE_NONNUMERIC, header=True)
    else:
        empty_file = open(path.join(path.dirname(path.realpath('__file__')), "..", "datasets",
                                    "stats_{}.csv".format(event_id)), "w")
        empty_file.close()


def process_events(q):
    while True:
        event = q.get()
        event_affected_code_lines = ast.literal_eval(event['affected_code_lines'])
        if str(event["_id"]) not in processed and event_affected_code_lines and event['dateTime'].year == 2017:
            event_year = event['dateTime'].year
            event_month = event['dateTime'].month
            event_hour = event['dateTime'].hour

            paths = get_file_paths(event_year, [event_month], [event_hour])
            print("Affected code lines: {}".format(event['affected_code_lines']))
            process_movto_files(paths, event['_id'], event['dateTime'], event_affected_code_lines,
                                event['lng'], event['lat'])
        q.task_done()


def parallelize(data, ref_function):
    data_split = np.array_split(data, partitions)
    data_split = [x for x in data_split if not x.empty]
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


if __name__ == '__main__':
    df_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                      "processed_tweets_affected_code_lines_1000.csv"), encoding='utf-8', sep=',',
                            dtype={'lat': str, 'lng': str})

    df_events["dateTime"] = pd.to_datetime(df_events.dateTime)

    df_events.index = df_events['dateTime']
    df_events = df_events.loc['2017-08']

    # df_exception_events_with_address = df_events.loc[(df_events['_id'] != 891724397370904576)]

    df_exception_events_with_address = df_events.loc[(df_events['label'] != "Irrelevant") &
                                                     (df_events['address'].notnull()) &
                                                     (df_events['lat'].notnull()) & (df_events['lng'].notnull())]

    # df_exception_events_with_address.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
    #                                                   "processed_tweets_affected_code_lines_100_august.csv"),
    #                                         sep=",", index=True, quoting=csv.QUOTE_NONNUMERIC, header=True)

    for i in range(num_fetch_threads):
        worker = Thread(target=process_events, args=(enclosure_queue,))
        worker.setDaemon(True)
        worker.start()

    for index, row in df_exception_events_with_address.iterrows():
        enclosure_queue.put(row)

    enclosure_queue.join()

    # df_exception_events_with_address.apply(lambda x: threads.append(MyThread(x).start()), axis=1)

