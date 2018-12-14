import pandas as pd
import glob
from os import path

events = ["accident", "natural_disaster", "social_event", "urban_event"]
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
distance = 1000


def csv_enricher(file_name):
    csv_data_frame = pd.read_csv(file_name)
    names = file_name.split("_")
    csv_data_frame["event_type"] = csv_data_frame.apply(lambda x: names[3].replace(".csv", ""), axis=1)
    csv_data_frame["event_id"] = csv_data_frame.apply(lambda x: names[2], axis=1)
    return csv_data_frame


year_view = {}
for month in months:
    print(month)
    for event in events:
        filenames = glob.glob(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                        "data_2017_{:02}_exception_events_{}m".format(month, distance),
                                        "apriori_velocities_*_{}.csv".format(event)))
        if filenames:
            combined_csv = pd.concat([csv_enricher(f) for f in filenames])
            file_path = path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                  "data_2017_{:02}_exception_events_{}m".format(month, distance),
                                  "combined_apriori_{}_{}.csv".format(event, month))
            combined_csv.to_csv(file_path, index=False)
            combined_data_frame = pd.read_csv(file_path, delimiter=",")
            combined_data_frame.columns = ["rule", "support", "confidence", "lift", "event_type", "event_id"]
            year_view[event] = (year_view.get(event, (0, 0, 0, 0, 0))[0] + len(filenames),
                                year_view.get(event, (0, 0, 0, 0, 0))[1] + len(combined_data_frame),
                                year_view.get(event, (0, 0, 0, 0, 0))[2] +
                                len(combined_data_frame.loc[(combined_data_frame.lift > 1) &
                                                            (combined_data_frame.support > 0.05)]),
                                year_view.get(event, (0, 0, 0, 0, 0))[3] + len(
                                    combined_data_frame.loc[combined_data_frame.lift == 1.0]),
                                year_view.get(event, (0, 0, 0, 0, 0))[4] + len(
                                    combined_data_frame.loc[(combined_data_frame.lift > 0) &
                                                            (combined_data_frame.lift < 1.0)]))

for event in events:
    print("\\textit{{{}}} & {} & {} & {} & {} & {} \\\\"
          .format(event, year_view[event][0], year_view[event][1], year_view[event][2], year_view[event][3],
                  year_view[event][4]))
