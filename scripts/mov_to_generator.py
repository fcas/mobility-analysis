import calendar


def get_file_paths(year, months, hours):
    file_paths = []
    if months is None:
        months = range(1, 13)
    if hours is None:
        hours = range(0, 24)

    for hour in hours:
        for month in months:
            weeks = calendar.monthcalendar(year, month)
            last_day = max(weeks[-1])
            for day in range(1, last_day + 1):
                event_hour = hour
                from_hour = "_".join(["Movto", "{}{:02d}{:02d}{:02d}00".format(
                    2017, month, day, event_hour)])
                if event_hour == 23:
                    if day != last_day:
                        to_hour = "{}{:02d}{:02d}{:02d}00".format(year, month, day + 1, 0)
                    else:
                        if month != 12:
                            to_hour = "{}{:02d}{:02d}{:02d}00".format(year, month + 1, 1, 0)
                        else:
                            to_hour = "{}{:02d}{:02d}{:02d}00".format(year + 1, 1, 1, 0)

                else:
                    to_hour = "{}{:02d}{:02d}{:02d}00".format(year, month, day, event_hour + 1)
                file_paths.append("_".join([from_hour, to_hour]))
    return file_paths
