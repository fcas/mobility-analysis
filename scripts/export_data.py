# -*- coding: utf-8 -*-

import tweets_processing_config
#from geolocation_config import google_geolocation_key
from tweets_processing_config import location_pattern, location_pattern_exception, address_pre_patterns, \
    address_padding, google_geolocation_bounds, google_geolocation_region, google_geolocation_url, \
    input_headers, address_anti_patterns, patterns_to_be_removed_pre_address, \
    patterns_to_be_removed_pos_address, address_pos_patterns, model_classes, google_geolocation_location_type
# from google.colab import files
from oauth2client.service_account import ServiceAccountCredentials
from numpy import *
from nltk.tokenize import TweetTokenizer

import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import itertools
import ast

import gspread
import csv
import pandas as pd
import redis
import os
import re
import requests
import unidecode

from os import path

import logging
logging.basicConfig(filename='tweets_analysis.log', level=logging.DEBUG)


r = redis.StrictRedis(host='localhost', port=6379, db=5)
seed = 42


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)


def load_tweets():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('project_config.json', scope)
    gc = gspread.authorize(credentials)

    rows = []
    for account_id in tweets_processing_config.ids:
        try:
            worksheet = gc.open(account_id).sheet1
            rows.extend(worksheet.get_all_values())
        except Exception as exception:
            logging.ERROR("Error processing {}".format(account_id), exception)
    csv_file = open(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "tweets",
                              'classified_tweets.csv'), 'w')
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    wr.writerows(rows)
    csv_file.close()


def save_raw_tweet(text):
    return text


def find_lat_long(address):
    global is_over_query_limit
    if not is_over_query_limit:
        cached_address = r.get(address)
        if cached_address is not None:
            cached_address = eval(cached_address.decode('utf-8'))
        if cached_address is None:
            url = google_geolocation_url \
                .format(address, google_geolocation_bounds, google_geolocation_region,
                        google_geolocation_location_type, google_geolocation_key)
            data = requests.get(url).json()
            is_over_query_limit = data.get("status") == "OVER_QUERY_LIMIT"
            if data.get("status") == "OK" and data.get("results", []):
                formatted_address = data["results"][0]["formatted_address"]
                lat = data["results"][0]["geometry"]["location"]["lat"]
                lng = data["results"][0]["geometry"]["location"]["lng"]
                location_type = data["results"][0]["geometry"]["location_type"]
                map_address[address] = (formatted_address, lat, lng, location_type)
                r.set(address, (formatted_address, lat, lng, location_type))
        else:
            map_address[address] = cached_address
    else:
        logging.ERROR("OVER_QUERY_LIMIT. Processing address: {}".format(address))


def find_address(text):
    extracted_address = ""

    if text:
        text = text + address_padding
        matches = re.finditer(location_pattern, text, re.IGNORECASE)

        try:
            extracted_address = next(matches, "").group()
        except AttributeError:
            pass

        if not extracted_address:
            matches = re.finditer(location_pattern_exception, text, re.IGNORECASE)
            try:
                extracted_address = next(matches, "").group()
            except AttributeError:
                pass

        if extracted_address:
            for address_pre_pattern in address_pre_patterns:
                extracted_address = re.compile(address_pre_pattern).split(extracted_address)[0]
            for address_pos_pattern in address_pos_patterns:
                extracted_address = re.compile(address_pos_pattern).split(extracted_address)[0]
            extracted_address = extracted_address.strip()
            extracted_address = extracted_address.lower()
            is_fake_pattern = False
            for anti_pattern in address_anti_patterns:
                if anti_pattern in extracted_address:
                    is_fake_pattern = True
            if is_fake_pattern or len(extracted_address.split(" ")) <= 1:
                return ""
            if not map_address.get(extracted_address):
                logging.info("extracted address: {}".format(extracted_address))
                found_addresses.add(extracted_address)
                find_lat_long(extracted_address)
                address_tokens.update(extracted_address.split(" "))

    return extracted_address


def standardize_text(df, patterns, text_field):
    for pattern in patterns:
        pattern = re.compile(pattern)
        df[text_field] = df[text_field].str.replace(pattern, "")
    df[text_field] = df[text_field].str.replace(r"  ", " ")
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.strip()
    return df


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_stop_words(tokens):
    for stopword in stopwords:
        tokens = list(filter(stopword.__ne__, tokens))
    return tokens


def stem_tokens(tokens):
    new_tokens = set()
    for token in tokens:
        if len(token) > 1:
            token = unidecode.unidecode(token)
            new_tokens.add(stemmer.stem(token))
    return list(new_tokens)


def tokens_to_text(tokens):
    return (" ".join(tokens)).strip()


def tokenize_tweets(tweets):
    tweet_tokenizer = TweetTokenizer()
    tweets["tokens"] = tweets["text"].apply(tweet_tokenizer.tokenize)
    tweets["tokens"] = tweets["tokens"].apply(remove_stop_words)
    tweets["tokens"] = tweets["tokens"].apply(stem_tokens)
    return tweets


if not os.path.isfile(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "tweets",
                                "classified_tweets.csv")):
    load_tweets()

df_raw_tweets = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "tweets",
                                      'classified_tweets.csv'), sep=',', dtype={'_id': str})
df_raw_tweets.columns = input_headers
df_raw_tweets["text"].fillna("", inplace=True)


if not os.path.isfile(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "tweets",
                                "processed_tweets.csv")):
    df_raw_tweets["raw_tweet"] = df_raw_tweets["text"].apply(save_raw_tweet)
    map_address = {}
    address_tokens = set()
    df_raw_tweets["location_type"] = ""
    found_addresses = set()
    is_over_query_limit = False

    vocabulary = None

    df_raw_tweets["text"] = df_raw_tweets["text"].apply(remove_emoji)
    df_raw_tweets = standardize_text(df_raw_tweets, patterns_to_be_removed_pre_address, "text")

    df_raw_tweets['address'] = df_raw_tweets['text'].apply(find_address)
    for key in list(map_address.keys()):
        df_raw_tweets.loc[df_raw_tweets['address'] ==
                          key, ['address', 'lat', 'lng', 'location_type']] = map_address.get(key)

    df_raw_tweets = standardize_text(df_raw_tweets, patterns_to_be_removed_pos_address, "text")

    stopwords = nltk.corpus.stopwords.words('portuguese')
    stopwords.extend(list(address_tokens))
    stemmer = nltk.stem.RSLPStemmer()

    tokenized_tweets = tokenize_tweets(df_raw_tweets)
    tokenized_tweets["text"] = tokenized_tweets["tokens"].apply(tokens_to_text)
    tokenized_tweets['dateTime'] = pd.to_datetime(tokenized_tweets.dateTime)
    tokenized_tweets.to_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "tweets",
                                      "processed_tweets.csv"), sep=",", index=False, quoting=csv.QUOTE_NONNUMERIC,
                            header=True)

tokenized_tweets = pd.read_csv(path.join(path.dirname(path.realpath(__file__)), "..", "datasets", "tweets",
                                         'processed_tweets.csv'), sep=',')
# cleaned tweets can be empty
tokenized_tweets["text"].fillna("", inplace=True)
tokenized_tweets = tokenized_tweets.loc[tokenized_tweets["text"] != ""]

# df_raw_tweets.to_csv("df_raw_tweets.csv", sep=",",
# index=False, quoting=csv.QUOTE_NONNUMERIC, header=True, encoding='utf-8')

# files.download('df_raw_tweets.csv')

"""# Corpus metrics without stopwords"""


def corpus_metrics(df):
    all_words = []
    sentence_lengths = []
    for tokens in df["tokens"].tolist():
        tokens = ast.literal_eval(tokens)
        if tokens:
            sentence_lengths.append(len(tokens))
            for token in tokens:
                all_words.append(token)
    global vocabulary
    vocabulary = sorted(list(set(all_words)))
    logging.info("%s words total, with a vocabulary size of %s" % (len(all_words), len(vocabulary)))
    logging.info("Max sentence length is %s" % max(sentence_lengths))

    plt.figure(figsize=(15, 15))
    plt.tick_params(labelsize=18)
    plt.xlabel('Sentence length', fontsize=20)
    plt.ylabel('Number of sentences', fontsize=20)
    plt.hist(sentence_lengths)
    plt.savefig(path.join(path.dirname(path.realpath(__file__)), "..", "..", "dissertacao", "images",
                          "corpus_metrics.png"))


corpus_metrics(tokenized_tweets)



list_corpus = tokenized_tweets["text"].tolist()
list_labels = tokenized_tweets["class_label"].tolist()


np.savetxt("data.csv", list_corpus, delimiter=",", fmt='%s')

np.savetxt("labels.csv", list_labels, delimiter=",", fmt='%s')

