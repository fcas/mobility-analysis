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
from apyori import apriori

# Set up projections
# WGS 84 (World Geodetic System) (aka WGS 1984, EPSG:4326)
from apriori_analysis import normalize_velocities
from mov_to_generator import get_file_paths


crs_wgs = proj.Proj(init='epsg:4326')
# Use a locally appropriate projected CRS (Coordinate Reference System)
crs_bng = proj.Proj(init='epsg:29101')
bus_position_to_event_distance = 100
events_lng_lat = set()

cores = cpu_count()
partitions = cores
processed = []

months = [8]
for m in months:
    data_path = path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                          "data_2017_{:02}_exception_events_{}m".format(m, bus_position_to_event_distance))
    if path.exists(data_path):
        for file in glob.glob(path.join(data_path, "apriori_*")):
            processed.append(file.split("velocities")[1].split("_")[1])

processed = list(set(processed))

num_fetch_threads = 1
enclosure_queue = Queue()
velocities = []


def velocity_status(df_affected_lines, event_id, event_date_time, event_affected_code_lines, event_lng, event_lat,
                    event_label, p):
    stats_frames = []

    df_stats = df_affected_lines.groupby('cd_linha')['nr_velocidade'].agg(['min', 'max', 'mean', 'median',
                                                                           'var', 'std', 'count',
                                                                           pd.np.count_nonzero])
    df_stats["nonzero_percentage"] = (1 - df_stats['count_nonzero'] / df_stats['count']) * 100
    df_stats["filename"] = p.split("Movto_")[1]
    df_stats['dateTime'] = event_date_time
    stats_frames.append(df_stats)

    if len(stats_frames) > 0:
        all_df = pd.concat(stats_frames)
        all_df.to_csv(path.join(path.dirname(path.realpath('__file__')),
                                "..", "datasets", "stats_{}_{}.csv".format(event_id, event_label)), sep=",", index=True,
                      quoting=csv.QUOTE_NONNUMERIC, header=True)
    else:
        empty_file = open(path.join(path.dirname(path.realpath('__file__')), "..", "datasets",
                                    "stats_{}_{}.csv".format(event_id, event_label)), "w")
        empty_file.close()


def process_movto_files(paths, event_id, event_date_time, event_affected_code_lines, event_lng, event_lat, event_label):
    print("Processing event id: {} ".format(event_id))
    print("Paths: {} ".format(paths))
    global velocities
    velocities = []
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
            df = pd.read_csv(io.BytesIO(data), sep=';', error_bad_lines=False)
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
                if not df_affected_lines.empty:
                    velocity_status(df_affected_lines, event_id, event_date_time, event_affected_code_lines, event_lng,
                                    event_lat, event_label, p)
                    df_affected_lines["dt_movto"] = pd.to_datetime(df_affected_lines.dt_movto)
                    df_affected_lines.index = df_affected_lines['dt_movto']

                    velocities.append(normalize_velocities(df_affected_lines))
            else:
                print("No records related to affected lines in {} file".format(p))

    with open(path.join(path.dirname(path.realpath('__file__')), "..", "datasets",
                        "velocities_{}_{}.csv".format(event_id, event_label)), 'w') as velocities_file:
        velocities_writer = csv.writer(velocities_file, delimiter=',', quoting=csv.QUOTE_NONE)
        for velocity_array in velocities:
            velocities_writer.writerow(velocity_array)

    csv_file = open(path.join(path.dirname(path.realpath('__file__')), "..", "datasets",
                              "apriori_velocities_{}_{}.csv".format(event_id, event_label)), 'w')
    apriori_writer = csv.writer(csv_file, delimiter=',')
    apriori_writer.writerow(["rule", "support", "confidence", "lift"])

    association_rules = apriori(velocities)
    association_results = list(association_rules)

    for item in association_results:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        apriori_writer.writerow(["{}, {}, {}, {}".format(" ".join(str(x) for x in items), str(item[1]),
                                                         str(item[2][0][2]), str(item[2][0][3]))])

    csv_file.close()


def process_events(q):
    while True:
        event = q.get()
        event_affected_code_lines = ast.literal_eval(event['affected_code_lines'])
        if event["_id"] not in processed and event_affected_code_lines and event['dateTime'].year == 2017:
            event_year = event['dateTime'].year
            event_month = event['dateTime'].month
            event_hour = event['dateTime'].hour
            event_weekday_name = event['dateTime'].weekday_name

            paths = get_file_paths(event_year, [event_month], [event_hour])
            paths = pd.DataFrame.from_records(paths)
            paths["dateTime"] = pd.to_datetime(paths.dateTime)
            paths.index = paths['dateTime']
            paths = paths.loc[paths.dateTime.dt.weekday_name == event_weekday_name]
            paths = paths.path.tolist()

            print("Affected code lines: {}".format(event['affected_code_lines']))
            process_movto_files(paths, event['_id'], event['dateTime'], event_affected_code_lines,
                                event['lng'], event['lat'], str(event['label']).lower().replace(" ", "_"))
        elif not event_affected_code_lines:
            print("No affected lines related to {} id".format(event["_id"]))
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
                                      "processed_tweets_affected_code_lines_{}.csv"
                                      .format(bus_position_to_event_distance)), encoding='utf-8', sep=',',
                            dtype={'lat': str, 'lng': str, '_id': str})

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

