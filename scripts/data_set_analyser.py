import os
import calendar

from mov_to_generator import get_file_paths

months_map = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
              9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
base_path = "/".join(['/Volumes', 'felipe'])
year = 2017


def get_data_set_description(paths):
    print("Missing files:\n")
    data_set_description = dict()
    for path in paths:
        file_path = "/".join([base_path, '{}.zip'.format(path)])
        result = os.path.isfile(file_path)
        if not result:
            print(file_path)
        if result:
            data_set_description[path[6:12]] = (data_set_description.get(path[6:12], (0, 0))[0] + 1,
                                                data_set_description.get(path[6:12], (0, 0))[1] +
                                                os.path.getsize(file_path))
    return data_set_description


def print_partial_latex_table(data_set_description):
    print("\nLatex table:\n")
    for month in range(1, 13):
        weeks = calendar.monthcalendar(year, int(month))
        last_day = max(weeks[-1])

        data_set_key = "{}{:02d}".format(year, month)
        data_set_description[data_set_key] = (data_set_description[data_set_key][0],
                                              data_set_description[data_set_key][1] / 1024 ** 3)

        print("\\hline\n {} & {} - {} & {} & {:.2f} \\\\".format(months_map.get(month), 1, last_day,
                                                                 data_set_description[data_set_key][0],
                                                                 data_set_description[data_set_key][1]))


if __name__ == '__main__':
    files = get_file_paths(year, None, None)
    description = get_data_set_description(files)
    print_partial_latex_table(description)

    total_size = 0
    total_files = 0
    for key in description.keys():
        total_files += description[key][0]
        total_size += description[key][1]
    print("\nTotal size: {:.02f}".format(total_size))
    print("\nTotal files: {}".format(total_files))
