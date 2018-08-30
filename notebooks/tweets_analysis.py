# -*- coding: utf-8 -*-

from notebooks import tweets_processing_config
from notebooks.geolocation_config import google_geolocation_key
from notebooks.tweets_processing_config import location_pattern, location_pattern_exception, address_pre_patterns, \
    address_padding, google_geolocation_bounds, google_geolocation_region, google_geolocation_url, \
    input_headers, address_anti_patterns, patterns_to_be_removed_pre_address, \
    patterns_to_be_removed_pos_address, address_pos_patterns
# from google.colab import files
from oauth2client.service_account import ServiceAccountCredentials
from numpy import *
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn import tree, svm
from sklearn.naive_bayes import GaussianNB


import nltk
import matplotlib.pyplot as plt


import numpy as np
import itertools

import gspread
import csv
import pandas as pd
import redis
import os
import re
import requests
import unidecode
import multiprocessing


r = redis.StrictRedis(host='localhost', port=6379, db=1)
cores = multiprocessing.cpu_count()

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
            print("Error processing {}".format(account_id), exception)
    csv_file = open('classified_tweets.csv', 'w')
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    wr.writerows(rows)
    csv_file.close()


def save_raw_tweet(text):
    return text


def find_lat_long(address):
    global is_over_query_limit
    if not is_over_query_limit:
        cached_address = r.get(address)
        if cached_address is None:
            url = google_geolocation_url \
                .format(address, google_geolocation_bounds, google_geolocation_region, google_geolocation_key)
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
            map_address[address] = eval(r.get(address).decode('utf-8'))


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
            if not map_address.get(extracted_address) and not is_fake_pattern:
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
    print(new_tokens)
    return list(new_tokens)


def tokens_to_text(tokens):
    return (" ".join(tokens)).strip()


def tokenize_tweets(tweets):
    tweet_tokenizer = TweetTokenizer()
    tweets["tokens"] = tweets["text"].apply(tweet_tokenizer.tokenize)
    tweets["tokens"] = tweets["tokens"].apply(remove_stop_words)
    tweets["tokens"] = tweets["tokens"].apply(stem_tokens)
    return tweets


if not os.path.isfile("classified_tweets.csv"):
    load_tweets()

df_raw_tweets = pd.read_csv('classified_tweets.csv', sep=',')
df_raw_tweets.columns = input_headers
df_raw_tweets["text"].fillna("", inplace=True)

if not os.path.isfile("processed_tweets.csv"):
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
    for found_address in found_addresses:
        print(found_address)
    print(map_address)
    print(address_tokens)
    for key in list(map_address.keys()):
        df_raw_tweets.loc[df_raw_tweets['address'] ==
                          key, ['address', 'lat', 'lng', 'location_type']] = map_address.get(key)

    df_raw_tweets = standardize_text(df_raw_tweets, patterns_to_be_removed_pos_address, "text")
    df_raw_tweets.head()

    stopwords = nltk.corpus.stopwords.words('portuguese')
    stopwords.extend(list(address_tokens))
    stemmer = nltk.stem.RSLPStemmer()

    print("Stopwords: ", stopwords)
    print("Tokens: ")

    tokenized_tweets = tokenize_tweets(df_raw_tweets)
    tokenized_tweets["text"] = tokenized_tweets["tokens"].apply(tokens_to_text)
    tokenized_tweets.to_csv("processed_tweets.csv", sep=",", index=True, quoting=csv.QUOTE_NONNUMERIC, header=True)

tokenized_tweets = pd.read_csv('processed_tweets.csv', sep=',')
# cleaned tweets can be empty
tokenized_tweets["text"].fillna("", inplace=True)
tokenized_tweets = tokenized_tweets.loc[tokenized_tweets["text"] != ""]

# df_raw_tweets.to_csv("df_raw_tweets.csv", sep=",",
# index=False, quoting=csv.QUOTE_NONNUMERIC, header=True, encoding='utf-8')

# files.download('df_raw_tweets.csv')

"""# Corpus metrics without stopwords"""


def corpus_metrics(df):
    all_words = [word for tokens in df["tokens"] for word in tokens]
    sentence_lengths = [len(tokens) for tokens in df["tokens"]]
    global vocabulary
    vocabulary = sorted(list(set(all_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_words), len(vocabulary)))
    print("Max sentence length is %s" % max(sentence_lengths))

    plt.figure(figsize=(15, 15))
    plt.tick_params(labelsize=18)
    plt.xlabel('Sentence length', fontsize=20)
    plt.ylabel('Number of sentences', fontsize=20)
    plt.hist(sentence_lengths)
    plt.show()


corpus_metrics(tokenized_tweets)

"""# **TFIDF**"""


def count_tokens_tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer


list_corpus = tokenized_tweets["text"].tolist()
list_labels = tokenized_tweets["class_label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.4, train_size=0.6)

print(X_train)
print(len(X_train))

# Applying TFIDF
X_train_tfidf, tfidf_vec = count_tokens_tfidf(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

print(tfidf_vec)


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


clf_logistic_regression = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                             multi_class='multinomial', n_jobs=cores, random_state=40)
clf_logistic_regression.fit(X_train_tfidf, y_train)

y_predicted_logistic_regression_clf = clf_logistic_regression.predict(X_test_tfidf)
print("Logistic Regression: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    get_metrics(y_test, y_predicted_logistic_regression_clf)))

"""# Decision Tree Classifier"""

decision_tree_clf = tree.DecisionTreeClassifier()
decision_tree_clf.fit(X_train_tfidf, y_train)

y_predicted_decision_tree_clf = decision_tree_clf.predict(X_test_tfidf)

print("Decision Tree: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    get_metrics(y_test, y_predicted_decision_tree_clf)))

"""# Naive Bayes"""

gnb_clf = GaussianNB()
gnb_clf.fit(X_train_tfidf.toarray(), y_train)


gnb_y_pred = gnb_clf.predict(X_test_tfidf.toarray())

print("Gaussian Naive Bayes: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    get_metrics(y_test, gnb_y_pred)))

"""# SVM"""

svm_clf = svm.SVC()
svm_clf.fit(X_train_tfidf, y_train)

y_predicted_svm_clf = svm_clf.predict(X_test_tfidf)

print("SVM: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
    get_metrics(y_test, y_predicted_svm_clf)))

"""# Validation"""


def plot_confusion_matrix(cm, classes, normalize=False, title='', cmap=plt.cm.winter):
    print(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tick_params(labelsize=18)
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt


# cm2 = confusion_matrix(y_test, y_predicted_decision_tree_clf)
# plt.figure(figsize=(13, 13))
# plot = plot_confusion_matrix(cm2, classes=model_classes, normalize=False)
# plt.show()


"""# Features"""


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
    plt.show()


importance_tfidf = get_most_important_features(tfidf_vec, clf_logistic_regression, 10)

top_scores = [a[0] for a in importance_tfidf[1]['tops']]
top_words = [a[1] for a in importance_tfidf[1]['tops']]
bottom_scores = [a[0] for a in importance_tfidf[1]['bottom']]
bottom_words = [a[1] for a in importance_tfidf[1]['bottom']]

plot_important_words(top_scores, top_words, bottom_scores, bottom_words)

print(importance_tfidf)
