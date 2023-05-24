import re
import csv
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def text_clustering():
    lista_tweets = []

    with open('datasets/dataset_textos_ptbr_2020.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        linha = 0
        for row in csv_reader:
            if linha == 0:
                linha = 1
            else:
                lista_tweets.append(row[1])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(lista_tweets)

    kmeans = KMeans(n_clusters=2).fit(tfidf)

    for tweet in lista_tweets:
        #text;cluster id
        print("{};{}".format(tweet.replace("\n", " "), kmeans.predict(tfidf_vectorizer.transform([tweet]))))

text_clustering()
