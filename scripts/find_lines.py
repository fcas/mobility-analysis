import config
import json
import csv
import os
from pymongo import MongoClient
from sptrans import SPTransClient

user = config.mongo["username"]
password = config.mongo["password"]
host = config.mongo["host"]
port = config.mongo["port"]
connection = MongoClient()

db = connection.gtfs_sptrans
root_dir = "/Volumes/felipe/Janeiro"

lines = []
trips = csv.reader(open(os.path.join(os.path.join(os.path.dirname(__file__)), 'gtfs', 'trips.txt')))
routes = csv.reader(open(os.path.join(os.path.join(os.path.dirname(__file__)), 'gtfs',  'routes.txt')))
trips_ids = set([line[0] for line in trips if "route_id" not in line])
route_ids = set([line[0] for line in routes if "route_id" not in line])

lines.extend(list(trips_ids))
lines.extend(list(route_ids))
lines.extend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
              "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "z", "w", "y"])

code_lines_not_found = open(os.path.join(root_dir, 'reports', 'code_lines_not_found.json'), 'w')
lines_not_found = set()

sp = SPTransClient()
sp.auth(config.sptrans["token"])

lines_map = {}

for line in lines:
    results = sp.search_by_bus(line)
    for result in results:
        code_line = result["CodigoLinha"]
        if db["lines"].find_one({"_id": code_line}) is None:
            result["_id"] = code_line
            db["lines"].insert_one(result)
        lines_map[result["CodigoLinha"]] = result


def get_row(row):
    row = row[0].split(";")
    line = int(row[1])
    line_info = {'CodigoLinha': line, 'Circular': '', 'Letreiro': '', 'Sentido': '', 'Tipo': '',
                 'DenominacaoTPTS': '', 'DenominacaoTSTP': '', 'Informacoes': ''}
    if line not in lines_map.keys():
        lines_not_found.add(line)
    else:
        line_info = lines_map.get(line)
    row.extend(list(line_info.values()))
    del row[1]
    row = ['' if value is None else str(value) for value in row]
    return row


for file in os.listdir(os.path.join(root_dir, "inputs")):
    rows = []
    mov_to = csv.reader(open(os.path.join(root_dir, "inputs",  file)))
    mov_to_list = list(mov_to)
    for mov in mov_to_list:
        rows.append(get_row(mov))
    writer = csv.writer(open(os.path.join(root_dir, "outputs",  file.replace("txt", "csv")), 'w'))
    writer.writerows(rows)

json.dump(list(lines_not_found), code_lines_not_found)
code_lines_not_found.close()
