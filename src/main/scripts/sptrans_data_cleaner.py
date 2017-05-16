import json

for i in range(0, 1):
    with open('sptrans' + str(i) + '.json', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for line in lines:
            data = json.loads(line)
            if "_id" in data:
                del data["_id"]
            if "mark" in data:
                mark = data["mark"]
                if mark is not None:
                    for key, value in mark.items():
                        data["mark_" + key] = value
                if "mark_ts" in data and "$date" in data["mark_ts"]:
                    data["mark_ts"] = data["mark_ts"]["$date"]
                del data["mark"]
            if "loc" in data:
                data["lat"] = data["loc"]["lat"]
                data["lng"] = data["loc"]["lng"]
                del data["loc"]
            if "ts" in data and "$date" in data["ts"]:
                data["ts"] = data["ts"]["$date"]
            if "bustrip_ts" in data and "$date" in data["bustrip_ts"]:
                data["bustrip_ts"] = data["bustrip_ts"]["$date"]
            line = json.dumps(data)
            f.write(line + '\n')