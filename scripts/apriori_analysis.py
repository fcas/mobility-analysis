import io
import sys
import zipfile
import numpy as np
import pandas as pd
import csv
import math

from os import path

from pandas.errors import ParserError

from mov_to_generator import get_file_paths
from apyori import apriori

freq = "300s"


def print_association_rules(rules):
    for rule in rules:
        # first index of the inner list
        # Contains base item and add item
        pair = rule[0]
        items = [x for x in pair]
        apriori_writer.writerow(["{}, {}, {}, {}".format(" ".join(items), str(rule[1]), str(rule[2][0][2]),
                                                         str(rule[2][0][3]))])


def normalize_velocities(df_movto):
    df_movto["dt_movto"] = pd.to_datetime(df_movto.dt_movto)
    df_movto.index = df_movto['dt_movto']
    grouped_ = df_movto.groupby(pd.Grouper(freq=freq))

    group_velocity = []
    for name, group in grouped_:
        velocity_median = 0
        if not group.empty:
            numbers = group["nr_velocidade"].tolist()
            nr_velocities = []
            for number in numbers:
                number = number if not math.isnan(number) else 0
                nr_velocities.append(number)
            velocity_median = np.average(nr_velocities)
        group_velocity.append(int(round(velocity_median)))
    return group_velocity


def process_velocities(month):
    paths = get_file_paths(2017, [month], None)
    paths = pd.DataFrame.from_records(paths)
    paths["dateTime"] = pd.to_datetime(paths.dateTime)
    paths.index = paths['dateTime']
    grouped_paths = paths.groupby(pd.Grouper(freq='D'))

    velocities = []
    for path_days in grouped_paths:
        paths = path_days[1].path.tolist()

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
                try:
                    df = pd.read_csv(io.BytesIO(data), sep=';')
                except ParserError as parser_error:
                    with open(path.join(path.dirname(path.realpath('__file__')), "..", "datasets",
                                        "bad_lines_{}.txt".format(p)), 'w') as fp:
                        sys.stderr = fp
                        print(parser_error)
                        df = pd.read_csv(io.BytesIO(data), sep=';', error_bad_lines=False)
                df.columns = [
                    "cd_evento_avl_movto", "cd_linha", "dt_movto", "nr_identificador", "nr_evento_linha", "nr_ponto",
                    "nr_velocidade", "nr_voltagem", "nr_temperatura_interna", "nr_evento_terminal_dado",
                    "nr_evento_es_1", "nr_latitude_grau", "nr_longitude_grau", "nr_indiceregistro", "dt_avl",
                    "nr_distancia", "cd_tipo_veiculo_geo", "cd_avl_conexao", "cd_prefixo"
                ]

                velocities.append(normalize_velocities(df))

    with open(path.join(path.dirname(path.realpath('__file__')), "..", "datasets",
                        "velocities_{}.csv".format(month)), 'w') as velocities_file:
        velocities_writer = csv.writer(velocities_file, delimiter=',', quoting=csv.QUOTE_NONE)
        for velocity_array in velocities:
            velocities_writer.writerow(velocity_array)
    return velocities


if __name__ == '__main__':
    months = [1]
    for m in months:
        try:
            v = process_velocities(m)
            csv_file = open(path.join(path.dirname(path.realpath('__file__')), "..", "datasets",
                                      "apriori_velocities_{}.csv".format(m)), 'w')
            apriori_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
            apriori_writer.writerow(["rule", "support", "confidence", "lift"])
            association_rules = apriori(v, min_length=2)
            association_results = list(association_rules)
            print_association_rules(association_results)
            csv_file.close()
        except Exception as e:
            print(e)
            pass
