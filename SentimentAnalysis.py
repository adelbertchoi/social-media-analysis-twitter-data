
"""
Assignment1: Code Part 5
by Adelbert Choi

This script aims to implement sentiment analysis of the twitter data obtained
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from colorama import Fore, Back, Style

import codecs
import tweetProcessor as tp
import pandas as pd
import matplotlib.pyplot as plt


def main():

    # tweets json filename
    jsonFilename = "uberTweetsUS.json"

    # returns tweetTokens and tweetDates
    tweets = tp.getTweetDf(jsonFilename, removeFreqWords=True)

    # get sentiments for each tweet
    vaderSentiments = vaderSentimentAnalysis(tweets["tweetTokens"], printSentiment=True)

    # preprate data for plotting
    tweet_df = pd.DataFrame({"Sentiments": vaderSentiments, "Date": tweets["tweetDates"]})
    tweet_df["Sentiments"] = tweet_df["Sentiments"].apply(pd.to_numeric)
    tweet_df["Date"] = tweet_df["Date"].apply(pd.to_datetime)
    tweet_df.set_index("Date", inplace=True)

    # plot histogram of sentiments
    plt.figure(figsize=(6, 5))
    plt.hist(tweet_df["Sentiments"], bins=20)
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

    # plot time series - change in mean hourly sentiments
    plt.figure(figsize=(6, 5))
    tweet_df.resample("1D").mean().plot()
    plt.ylabel("Mean hourly sentiment")
    plt.xlabel("Time")
    plt.show()

    # preprate data to make a barchart showing daily proportion of negative tweets
    # classify whether a tweet is negative or not
    tweet_df.loc[tweet_df["Sentiments"] < 0, "NegativeTweet"] = 1
    tweet_df.loc[tweet_df["Sentiments"] >= 0, "NegativeTweet"] = 0

    propNegativeTweetsDay = tweet_df["NegativeTweet"].resample("1D").mean()
    xTicks = tweets["tweetDates"].apply(pd.to_datetime)
    xTicks = xTicks.dt.strftime('%d-%m-%y').unique()
    xTicks = xTicks[::-1]

    # plot bar chart showing the changes in daily proportion of negative tweets
    plt.bar(range(len(propNegativeTweetsDay)), propNegativeTweetsDay, tick_label=xTicks)
    plt.ylabel("Daily proportion of negative tweets")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.show()



def vaderSentimentAnalysis(tweetTokens, printSentiment=False):
    """
    A method that utilised vader sentiment analysis to compute for the sentiment score for each tweet
    Note: method obtained from Lecturer Jeffrey Chan

    :param tweetTokens:
    :param printSentiment: whether to print the sentiments computed for each tweet
    :return: a list of sentiment scores
    """

    # set sentiment analysis
    sentAnalyser = SentimentIntensityAnalyzer()

    # an initial list to store sentiment scores
    lSentiment = []

    for tokens in tweetTokens:
        # compute the sentiment scores (or polarity score)
        dSentimentScores = sentAnalyser.polarity_scores(" ".join(tokens))
        lSentiment.append(dSentimentScores['compound'])

        if printSentiment == True:
            for cat, score in dSentimentScores.items():
                print('{0}: {1}, '.format(cat, score), end='')
            print()

    return lSentiment

# the following methods were utilised initially
# however, I excluded results from analysis since they are consistent with results from vader sentiment analysis
def countWordSentimentAnalysis(tweetTokens, printSentiment=False):

    # load set of positive words
    lPosWords = []
    with open("positive-words.txt", 'r', encoding='utf-8', errors='ignore') as fPos:
        for sLine in fPos:
            lPosWords.append(sLine.strip())

    setPosWords = set(lPosWords)

    # load set of negative words
    lNegWords = []
    with codecs.open("negative-words.txt", 'r', encoding='utf-8', errors='ignore') as fNeg:
        for sLine in fNeg:
            lNegWords.append(sLine.strip())

    setNegWords = set(lNegWords)

    lSentiment = []
    for tweet in tweetTokens:
        # compute the sentiment
        sentiment = computeSentiment(tweet, setPosWords, setNegWords)
        lSentiment.append(sentiment)

        # if we are printing, each token is printed and coloured according to red if positive word, and blue
        if printSentiment == True:
            for word in tweet:
                if word in setPosWords:
                    print(Fore.RED + word + ', ', end='')
                elif word in setNegWords:
                    print(Fore.BLUE + word + ', ', end='')
                else:
                    print(Style.RESET_ALL + word + ', ', end='')
            print(': {}'.format(sentiment))

    return lSentiment

def computeSentiment(tweetTokens, positiveWords, negativeWords):

    posNum = len([word for word in tweetTokens if word in positiveWords])
    negNum = len([word for word in tweetTokens if word in negativeWords])

    # replace the right hand side with how to compute the sentiment value
    sentimentVal = posNum - negNum

    return sentimentVal;



if __name__ == '__main__':
    main()