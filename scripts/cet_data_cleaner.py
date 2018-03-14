import json
import datetime

with open('cet.json', 'r+') as f:
    lines = f.readlines()
    f.seek(0)
    f.truncate()
    for line in lines:
        data = json.loads(line)
        date = data["dateTime"]
        data["dateTime"] = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").isoformat()
        line = json.dumps(data)
        f.write(line + '\n')