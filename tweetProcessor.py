
"""
Assignment1: Code Part 2
by Adelbert Choi

This script contains code involving text data pre-processing
"""

from collections import Counter
from nltk.corpus import wordnet

import pandas as pd
import json
import string
import nltk
import re


def getTweetDf(jsonFilename, type="sentiment", removeFreqWords=False):
    """
    A method to return a data frame containing the text pre-processed tweets and their corresponsing tweet date
    Note: method adapted from Lecturer Jeffrey Chan

    :param jsonFilename:
    :param type="sentiment" - returns a df object with tweets ready for sentiment analysis
           type="topic" - returns a df object with tweets ready for topic modelling:
    :param removeFreqWords: additional parameter to remove other identified frequently occurring words
    :return: a data frame
    """

    # save tweets and their dates in these lists
    tweetTokens = []
    tweetDates = []

    # open json and pre-process each tweet
    with open(jsonFilename, 'r') as file:
        for line in file:
            tweet = json.loads(line).get('text', '')

            # tokenise, filter stopwords and convert words to lower case
            tokens = processTweet(tweet, removeFreqWords)

            if type == "sentiment":
                tweetTokens.append(tokens)
            if type == "topic":
                tweetTokens.append(' '.join(tokens))

            tweetDates.append(json.loads(line).get('created_at'))

    return pd.DataFrame({"tweetTokens": tweetTokens, "tweetDates": tweetDates})

#####
def getWordPOS(POStag):
    """
    A method that returns a part of speech tag for an certain word
    this is used to assist during lemmatisation
    Method code obtained from https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python

    :param POStag -  a part of speech tag associated with a certain word:
    :return: identified part of speech tag
    """

    if POStag.startswith('J'):
        return wordnet.ADJ
    elif POStag.startswith('R'):
        return wordnet.ADV
    elif POStag.startswith('N'):
        return wordnet.NOUN
    elif POStag.startswith('V'):
        return wordnet.VERB
    else:
        return wordnet.NOUN



#####
def processTweet(tweet, removeFreqWords=False):
    """
    A method to pre-process a certain tweet
    Note: method adapted from Lecturer Jeffrey Chan

    :param tweet:
    :param removeFreqWords: additional parameter to remove other identified frequently occurring words
    :return: pre-processed tweet
    """

    # tweet tokeniser
    tweetTokeniser = nltk.tokenize.TweetTokenizer()
    # tweet lemmatizer
    tweetLemmatizer = nltk.stem.WordNetLemmatizer()

    # list of stopwords
    Stopwords = nltk.corpus.stopwords.words('english')  # english stopwords
    Stopwords = Stopwords + list(string.punctuation)
    Stopwords = Stopwords + ["u", "w", "b", "n", "—", "la", "ur"]
    Stopwords = Stopwords + ['rt', 'via', '...', '…', '’', '“', '”', '', '..', "️", "‍"]
    Stopwords = Stopwords + [".", "’", "…", "!", "?", "...", ":", "“", "”", "$", '"', "/", "-", ".."]
    Stopwords = Stopwords + ["(", ")", "*", "&"]

    # some patterns that are not desired
    # pattern to remove all strings of digits or fractions, e.g., 6.15
    regexDigit = re.compile("^\d+\s|\s\d+\s|\s\d+$")
    # regex pattern for http
    regexHttp = re.compile("http")

    # if removeFreqWords==True then we will include frequent words also as stopwords
    if removeFreqWords == True:
        Stopwords = Stopwords + ["uber", "driver", "get"]

    # start text pre-processing
    # covert all to lower case
    tweet = tweet.lower()
    # tokenise tweets
    tweetTokens = tweetTokeniser.tokenize(tweet)
    # remove white spaces
    tweetTokens = [word.strip() for word in tweetTokens]

    # irrelevant word removal
    tweetTokens = [word for word in tweetTokens if word not in Stopwords and not word.isdigit() and regexHttp.match(word) == None and regexDigit.match(word) == None]

    # obtain part of speech tag for each word in tweetTokens
    tweetTokens_pos = nltk.pos_tag(tweetTokens)
    # lemmatise the words
    lemmedTokens = set([tweetLemmatizer.lemmatize(word[0], getWordPOS(word[1])) for word in tweetTokens_pos])

    return lemmedTokens


#####
def getFrequentTerms(tweetTokens, printTerms=True, freqNum=50):
    """
    A method that returns a comprehensive count of each term present across all tweets
    Note: method adapted from Lecturer Jeffrey Chan

    :param tweetTokens:
    :param printTerms: whether to print the frequent terms or not
    :param freqNum: determines top frequent terms to print
    :return: a counter object
    """

    # term frequency counter
    termFreqCounter = Counter()

    for tokens in tweetTokens:
        # update count
        termFreqCounter.update(tokens)

    if printTerms == True:
        # print out most common terms
        for term, count in termFreqCounter.most_common(freqNum):
            print('"' + term + '"' + ' : ' + str(count))

    return termFreqCounter


#####
def getHashtags(tweet):
    """
    A method to obtain the hashtags in a certain tweet
    Note: method obtained from Lecturer Jeffrey Chan

    :param tweet:
    :return: list of hashtags (in lower case)
    """

    entities = tweet.get('entities', {})
    hashtags = entities.get('hashtags', [])

    return [tag["text"].lower() for tag in hashtags]


#####
def getFrequentHastags(jsonFilename, printTerms=True, tweetThres=50):
    """
     A method that returns a comprehensive count of each hastag present across all tweets
     Note: method adapted from Lecturer Jeffrey Chan

     :param tweetTokens:
     :param printTerms: whether to print the frequent terms or not
     :param tweetThres: determines top frequent hastags to print
     :return: a counter object
     """

    # hashtag frequency counter
    hashtagsCounter = Counter()

    with open(jsonFilename, 'r') as file:
        for line in file:
            hashtagsInTweet = getHashtags(json.loads(line))
            hashtagsCounter.update(hashtagsInTweet)

        if printTerms == True:
            for tag, count in hashtagsCounter.most_common(tweetThres):
                print(tag + ": " + str(count))

    return hashtagsCounter