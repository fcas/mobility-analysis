# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import tweets_processing_config
from geolocation_config import google_geolocation_key
from tweets_processing_config import location_pattern, location_pattern_exception, address_pre_patterns, \
    address_padding, google_geolocation_bounds, google_geolocation_region, google_geolocation_url, \
    input_headers, address_anti_patterns, patterns_to_be_removed_pre_address, \
    patterns_to_be_removed_pos_address, address_pos_patterns, model_classes, google_geolocation_location_type
# from google.colab import files
from oauth2client.service_account import ServiceAccountCredentials
from numpy import *
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, make_scorer

from sklearn import tree, svm, neighbors, model_selection
from sklearn.naive_bayes import MultinomialNB, ComplementNB

import nltk
import matplotlib.pyplot as plt


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
    pd.set_option('display.height', 1000)
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

"""# **TFIDF**"""


def count_tokens_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer


list_corpus = tokenized_tweets["text"].tolist()
list_labels = tokenized_tweets["class_label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, random_state=seed,
                                                    test_size=0.4, train_size=0.6)

# Applying TFIDF
X_train_tfidf_counts, tfidf_vec = count_tokens_tfidf(X_train)
X_test_tfidf_counts = tfidf_vec.transform(X_test)


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar().ax.tick_params(labelsize=20)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.tick_params(labelsize=18)
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.tight_layout()

    return plt


def plot_important_words(t_scores, t_words, b_scores, b_words):
    y_pos = np.arange(len(t_words))
    top_pairs = [(a, b) for a, b in zip(t_words, t_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a, b) for a, b in zip(b_words, b_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    t_words = [a[0] for a in top_pairs]
    t_scores = [a[1] for a in top_pairs]

    b_words = [a[0] for a in bottom_pairs]
    b_scores = [a[1] for a in bottom_pairs]

    plt.figure(figsize=(10, 10))

    plt.subplot(121)
    plt.barh(y_pos, b_scores, align='center', alpha=0.5)
    plt.title('Irrelevante', fontsize=20)
    plt.yticks(y_pos, b_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplot(122)
    plt.barh(y_pos, t_scores, align='center', alpha=0.5)
    plt.title('Evento de Exceção', fontsize=20)
    plt.yticks(y_pos, t_words, fontsize=14)
    plt.suptitle("Most important words for relevance", fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplots_adjust(wspace=0.8)
    return plt


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importance = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coefficient = sorted(word_importance, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coefficient[:n], key=lambda x: x[0])
        bottom = sorted_coefficient[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes


metrics = {}
names = []
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, pos_label=None, average='weighted'),
           'recall': make_scorer(recall_score, pos_label=None, average='weighted'),
           'f1_score': make_scorer(f1_score, pos_label=None, average='weighted')}

np.random.seed(seed)

models = []

logistic_regression_clf = LogisticRegression()
logistic_regression_clf.random_state = seed
models.append(("lr", logistic_regression_clf))

decision_tree_clf = tree.DecisionTreeClassifier()
decision_tree_clf.random_state = seed
models.append(("dt", decision_tree_clf))

mnb_clf = MultinomialNB()
mnb_clf.random_state = seed
models.append(("mnb", mnb_clf))

cnb_clf = ComplementNB()
cnb_clf.random_state = seed
models.append(("cnb", cnb_clf))

svm_clf = svm.SVC()
svm_clf.random_state = seed
models.append(("svm", svm_clf))

random_forest_clf = RandomForestClassifier()
random_forest_clf.random_state = seed
models.append(("rf", random_forest_clf))

knn_clf = neighbors.KNeighborsClassifier()
knn_clf.random_state = seed
models.append(("knn", knn_clf))

mlp_clf = MLPClassifier()
mlp_clf.random_state = seed
models.append(("mlp", mlp_clf))


for name, model in models:
    k_fold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_validate(model, X_train_tfidf_counts, y_train, cv=k_fold, scoring=scoring,
                                                return_train_score=False, n_jobs=4)
    y_pred = model_selection.cross_val_predict(model, X_train_tfidf_counts, y_train, cv=k_fold, n_jobs=4)

    cm_decision_tree = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(13, 13))
    plt = plot_confusion_matrix(cm_decision_tree, classes=model_classes, normalize=True, title='')
    plt.savefig(path.join(path.dirname(path.realpath(__file__)), "..", "..", "dissertacao", "images",
                          "confusion_matrix_{}.png".format(name)))

    importance_tfidf = get_most_important_features(tfidf_vec, model, 10)
    top_scores = [a[0] for a in importance_tfidf[1]['tops']]
    top_words = [a[1] for a in importance_tfidf[1]['tops']]
    bottom_scores = [a[0] for a in importance_tfidf[1]['bottom']]
    bottom_words = [a[1] for a in importance_tfidf[1]['bottom']]

    plt = plot_important_words(top_scores, top_words, bottom_scores, bottom_words)
    plt.savefig(path.join(path.dirname(path.realpath(__file__)), "..", "..", "dissertacao", "images",
                          "important_words_{}.png".format(name)))

    names.append(name)
    for key_result in sort(list(cv_results.keys())):
        if metrics.get(key_result, []):
            results = metrics.get(key_result)
            results.append(cv_results[key_result])
        else:
            metrics[key_result] = [cv_results[key_result]]

        print("{} {} {} {}".format(name, key_result, np.mean(cv_results[key_result]), np.std(cv_results[key_result])))

for metric_key in metrics.keys():
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(metrics[metric_key])
    ax.set_xticklabels(names)
    plt.savefig(path.join(path.dirname(path.realpath(__file__)), "..", "..", "dissertacao", "images",
                          "box_plot_{}.png".format(metric_key)))

