#!/usr/bin/env python
from pymongo import GEOSPHERE
from pymongo import MongoClient
from src.main.scripts import config
from datetime import datetime

user = config.mongo["username"]
password = config.mongo["password"]
host = config.mongo["host"]
port = config.mongo["port"]
connection = MongoClient()

db = connection.gtfs_sptrans


def persist_shapes():
    file = open("../resources/gtfs_sptrans/shapes.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                document = \
                    {
                        "shape_id": values[0].replace("\"", ""),
                        "shape_pt_sequence": int(values[3].replace("\"", "")),
                        "shape_dist_traveled": float(values[4].replace("\"", "")),
                        "location": {
                            "type": "Point",
                            "coordinates": [
                                float(values[2].replace("\"", "")),
                                float(values[1].replace("\"", ""))
                            ]
                        }
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["shapes"].insert_one(document)
        i += 1

    db["shapes"].create_index([("location", GEOSPHERE)])


def persist_stops():
    file = open("../resources/gtfs_sptrans/stops.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.replace(", ", "*").split(",")
            document = {}
            try:
                document = \
                    {
                        "stop_id": values[0].replace("\"", ""),
                        "stop_name": values[1].replace("*", ", ").replace("\"", ""),
                        "stop_desc": values[2].replace("\"", ""),
                        "location": {
                            "type": "Point",
                            "coordinates": [
                                float(values[4].replace("\"", "")),
                                float(values[3].replace("\"", ""))
                            ]
                        }
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["stops"].insert_one(document)
        i += 1

    db["stops"].create_index([("location", GEOSPHERE)])


def persist_trips():
    file = open("../resources/gtfs_sptrans/trips.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                document = \
                    {
                        "route_id": values[0].replace("\"", ""),
                        "service_id": values[1].replace("\"", ""),
                        "trip_id": values[2].replace("\"", ""),
                        "trip_headsign": values[3].replace("\"", ""),
                        "direction_id": values[4].replace("\"", ""),
                        "shape_id": values[5].replace("\"", "").replace("\n", "")
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["trips"].insert_one(document)
        i += 1


def persist_stop_times():
    file = open("../resources/gtfs_sptrans/stop_times.txt", "r")
    i = 0
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                arrival_time_str = values[1].replace("\"", "")
                arrival_hour = arrival_time_str[:2]
                if arrival_hour == "24":
                    arrival_time_str = str().join(["00", arrival_time_str[2:]])
                arrival_time = datetime.strptime(arrival_time_str, '%H:%M:%S').time()

                departure_time_str = values[2].replace("\"", "")
                departure_hour = departure_time_str[:2]
                if departure_hour == "24":
                    departure_time_str = str().join(["00", departure_time_str[2:]])
                departure_time = datetime.strptime(departure_time_str, '%H:%M:%S').time()

                document = \
                    {
                        "trip_id": values[0].replace("\"", ""),
                        "arrival_time": (now.replace(hour=arrival_time.hour, minute=arrival_time.minute,
                                                     second=arrival_time.second) - midnight).total_seconds(),
                        "departure_time":
                            (now.replace(hour=departure_time.hour, minute=departure_time.minute,
                                         second=departure_time.second) - midnight).total_seconds(),
                        "stop_id": values[3].replace("\"", ""),
                        "stop_sequence": values[4].replace("\"", "")
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["stop_times"].insert_one(document)
        i += 1


def persist_routes():
    file = open("../resources/gtfs_sptrans/routes.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                document = \
                    {
                        "route_id": values[0].replace("\"", ""),
                        "agency_id": values[1].replace("\"", ""),
                        "route_short_name": values[2].replace("\"", ""),
                        "route_long_name": values[3].replace("\"", ""),
                        "route_type": values[4].replace("\"", ""),
                        "route_color": values[5].replace("\"", ""),
                        "route_text_color": values[6].replace("\"", "").replace("\n","")
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["routes"].insert_one(document)
        i += 1


def persist_frequencies():
    file = open("../resources/gtfs_sptrans/frequencies.txt", "r")
    i = 0
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                start_time_str = values[1].replace("\"", "")
                start_hour = start_time_str[:2]
                if start_hour == "24":
                    start_time_str = str().join(["00", start_time_str[2:]])
                start_time = datetime.strptime(start_time_str, '%H:%M:%S').time()

                end_time_str = values[2].replace("\"", "")
                end_hour = end_time_str[:2]
                if end_hour == "24":
                    end_time_str = str().join(["00", end_time_str[2:]])
                end_time = datetime.strptime(end_time_str, '%H:%M:%S').time()

                document = \
                    {
                        "trip_id": values[0].replace("\"", ""),
                        "start_time": (now.replace(hour=start_time.hour, minute=start_time.minute,
                                                   second=start_time.second) - midnight).total_seconds(),
                        "end_time": (now.replace(hour=end_time.hour, minute=end_time.minute,
                                                 second=end_time.second) - midnight).total_seconds(),
                        "headway_secs": float(values[3].replace("\"", "").replace("\n", ""))
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["frequencies"].insert_one(document)
        i += 1


def persist_fare_rules():
    file = open("../resources/gtfs_sptrans/fare_rules.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                document = \
                    {
                        "fare_id": values[0].replace("\"", ""),
                        "route_id": values[1].replace("\"", ""),
                        "origin_id": values[2].replace("\"", ""),
                        "destination_id": values[3].replace("\"", ""),
                        "contains_id": values[4].replace("\"", "").replace("\n", "")
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["fare_rules"].insert_one(document)
        i += 1


def persist_fare_attributes():
    file = open("../resources/gtfs_sptrans/fare_attributes.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                document = \
                    {
                        "fare_id": values[0].replace("\"", ""),
                        "price": values[1].replace("\"", ""),
                        "currency_type": values[2].replace("\"", ""),
                        "payment_method": values[3].replace("\"", ""),
                        "transfers": values[4].replace("\"", ""),
                        "transfer_duration": values[5].replace("\"", "").replace("\n", "")
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["fare_attributes"].insert_one(document)
        i += 1


def persist_calendar():
    file = open("../resources/gtfs_sptrans/calendar.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                document = \
                    {
                        "service_id": values[0].replace("\"", ""),
                        "monday": values[1].replace("\"", ""),
                        "tuesday": values[2].replace("\"", ""),
                        "wednesday": values[3].replace("\"", ""),
                        "thursday": values[4].replace("\"", ""),
                        "friday": values[5].replace("\"", ""),
                        "saturday": values[6].replace("\"", ""),
                        "sunday": values[7].replace("\"", ""),
                        "start_date": datetime.strptime(values[8].replace("\"", ""), '%Y%M%d').timestamp(),
                        "end_date": datetime.strptime(
                            values[9].replace("\"", "").replace("\n", ""), '%Y%M%d').timestamp()
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["calendar"].insert_one(document)
        i += 1


def persist_agency():
    file = open("../resources/gtfs_sptrans/agency.txt", "r")
    i = 0
    for line in file:
        if i > 0:
            values = line.split(",")
            document = {}
            try:
                document = \
                    {
                        "agency_id": values[0].replace("\"", ""),
                        "agency_name": values[1].replace("\"", ""),
                        "agency_url": values[2].replace("\"", ""),
                        "agency_timezone": values[3].replace("\"", ""),
                        "agency_lang": values[4].replace("\"", "")
                    }
            except Exception as e:
                print(line, e)

            if document:
                db["agency"].insert_one(document)
        i += 1


def main():
    persist_shapes()
    persist_calendar()
    persist_stops()
    persist_agency()
    persist_trips()
    persist_fare_attributes()
    persist_fare_rules()
    persist_frequencies()
    persist_routes()
    persist_stop_times()

if __name__ == '__main__':
    main()
