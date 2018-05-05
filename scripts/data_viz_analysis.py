import pandas as pd
import matplotlib.pyplot as plt
from os import path

df_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "affected_lines.csv"),
                        encoding='utf-8', sep=';')

plt.figure(figsize=(24, 16))
plt.scatter(x=df_events['code_line'], y=df_events['qtd_exception_events'], alpha=0.6)
plt.xlabel('Quantity of exception events', fontsize=22)
plt.ylabel('Bus line codes', fontsize=22)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.grid(True)
plt.show()


processed_events = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets",
                                         "processed_tweets_CETSP_.csv"), encoding='utf-8', sep=';')
processed_events = processed_events.loc[(processed_events['label'] != "Irrelevant") &
                                        (processed_events['address'].notnull()) &
                                        (processed_events['lat'].notnull()) & (processed_events['lng'].notnull())]


processed_events["address"] = processed_events.apply(lambda x: x["address"].split(",")[0].split("-")[0], axis=1)
processed_events = processed_events.groupby('address')['address'].count()

processed_events = processed_events.to_frame()
processed_events = processed_events.sort_values(by=['address'], ascending=False)

index = list(range(0, len(processed_events.address[0:50])))
plt.figure(figsize=(24, 16))
plt.bar(index, processed_events.address[0:50], align='center')
plt.xticks(index, processed_events.index.values[0:50], rotation='vertical', fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.show()

pt = pd.pivot_table(df_events, values=["qtd_exception_events"], index=['code_line', 'identification'])
pt.sort_values(by='qtd_exception_events', ascending=False, inplace=True)
print(pt)
