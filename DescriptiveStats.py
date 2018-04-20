
"""
Assignment1: Code Part 3
by Adelbert Choi

This script aims to generate simple descriptive summaries of the data
"""

import tweetProcessor as tp
import matplotlib.pyplot as plt

def main():

    """
    This script mainly utilised tweetProcessor to obtained a pre-processed data frame of the tweets
    Also, descriptive summaries of the tweets were also obtained

    """

    # tweets json filename
    jsonFilename = "uberTweetsUS.json"

    # returns tweetTokens and tweetDates
    # removeFreqWords=False to return a comprehensive list of frequent terms
    tweets = tp.getTweetDf(jsonFilename, removeFreqWords=False)
    # get frequent terms
    termsCount = tp.getFrequentTerms(tweets["tweetTokens"])
    # get frequent hastags
    hashtagsCount = tp.getFrequentHastags(jsonFilename)

    # preprare data for bar chart of frequent terms
    terms = [term for term, count in termsCount.most_common(30)]
    tCounts = [count for tag, count in termsCount.most_common(30)]
    terms.reverse()
    tCounts.reverse()

    # plot bar chat of frequent terms
    plt.figure(figsize=(6, 5))
    plt.barh(range(len(tCounts)), tCounts, tick_label=terms)
    plt.ylabel("Number of tweets with term frequency")
    plt.xlabel("Term frequency")
    plt.show()

    # preprare data for bar chat of frequent terms
    hastags = [term for term, count in hashtagsCount.most_common(30)]
    hCounts = [count for tag, count in hashtagsCount.most_common(30)]
    hastags.reverse()
    hCounts.reverse()

    # plot bar chat of frequent terms
    plt.figure(figsize=(6, 5))
    plt.barh(range(len(hCounts)), hCounts, tick_label=hastags)
    plt.ylabel("Number of tweets with hashtag frequency")
    plt.xlabel("Hastag frequency")
    plt.show()


if __name__ == '__main__':
    main()