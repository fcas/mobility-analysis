import pandas as pd
import glob
from os import path

events = ["accident", "natural_disaster", "social_event", "urban_event"]
months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november",  "december"]


def csv_enricher(file_name):
    csv_data_frame = pd.read_csv(file_name)
    names = file_name.split("_")
    csv_data_frame["event_type"] = csv_data_frame.apply(lambda x: names[3].replace(".csv", ""), axis=1)
    csv_data_frame["event_id"] = csv_data_frame.apply(lambda x: names[2], axis=1)
    return csv_data_frame


for month in months:
    print(month)
    for event in events:
        filenames = glob.glob(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                        "apriori_2017_{}_exception_events".format(month),
                                        "apriori_velocities_*_{}.csv".format(event)))
        if filenames:
            combined_csv = pd.concat([csv_enricher(f) for f in filenames])
            file_path = path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                  "apriori_2017_{}_exception_events".format(month),
                                  "combined_apriori_{}_{}.csv".format(event, month))
            combined_csv.to_csv(file_path, index=False)
            combined_data_frame = pd.read_csv(file_path, delimiter=",")
            combined_data_frame.columns = ["rule", "support", "confidence", "lift", "event_type", "event_id"]
            print("\\textit{{{}}} & {} & {} & {} & {} & {} \\\\"
                  .format(event, len(filenames), len(combined_data_frame),
                          len(combined_data_frame.loc[(combined_data_frame.lift > 1) &
                                                      (combined_data_frame.support > 0.05)]),
                          len(combined_data_frame.loc[combined_data_frame.lift == 1.0]),
                          len(combined_data_frame.loc[(combined_data_frame.lift > 0) &
                                                      (combined_data_frame.lift < 1.0)])))
