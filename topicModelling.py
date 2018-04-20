
"""
Assignment1: Code Part 4
by Adelbert Choi

This script aims to implement topic modeling on the twitter data obtained
"""

import math
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from numpy.random import seed
from wordcloud import WordCloud
from ggplot import *

import SentimentAnalysis as sa
import tweetProcessor as tp
import matplotlib.pyplot as plt
import pandas as pd



def main():
    """
    This method runs topic modelling through LDA.
    Note: code adapted from Lecturer Jeffrey Chan

    :return:
    """
    # tweets json filename
    jsonFilename = "uberTweetsUS.json"

    # returns tweetTokens and tweetDates
    tweets = tp.getTweetDf(jsonFilename, type="topic", removeFreqWords=True)


    featureNum = 250        # this is the number of features/words to used to describe our documents
    wordNumToDisplay = 20   # number of words to display for each topic
    topicNum = 3            # number of topics to be created

    # Count Vectorizer
    tfVectorizer = CountVectorizer(max_df=0.95, min_df=10, max_features=featureNum, lowercase=False, stop_words=None)
    # Create a term document matrix
    tf = tfVectorizer.fit_transform(tweets["tweetTokens"])

    # Extract the names of the features - words
    tfFeatureNames = tfVectorizer.get_feature_names()

    # Set seed to allow reproducibility of results
    seed(7777)
    # Implement topic modeling using LDA
    ldaModel = LatentDirichletAllocation(n_components=topicNum, max_iter=10, learning_method='online').fit(tf)

    # Print out topics
    display_topics(ldaModel, tfFeatureNames, wordNumToDisplay)


    ### The following set of codes were adapted from https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
    # The codes below aims to assign a topic to each tweet based on the constructed topic model
    # also the overall topic distribution is also obtained

    # Obtain ldaModel output
    lda_output = ldaModel.transform(tf)

    topicNames = ["Topic" + str(i) for i in range(topicNum)]                    # topic names e.g., Topic 0, 1, ..
    tweetNames = ["Tweet" + str(i) for i in range(len(tweets["tweetTokens"]))]  # tweet names e.g., Tweet 0, 1, ..

    # Make a pandas dataframe
    # this dataframe has assigned probabilities that a certain tweet is topic, 0, 1, or 2
    tweets_and_topics = pd.DataFrame(np.round(lda_output, 2), columns=topicNames, index=tweetNames)

    # Get dominant topic for each tweet
    # Return topic for a certain tweet if probability to a certain topic is the highest
    tweet_dominant_topic = np.argmax(tweets_and_topics.values, axis=1)
    tweets_and_topics["dominant_topic"] = tweet_dominant_topic

    # Print Overall Topic Distribution
    print("Topic Distribution")
    df_topic_distribution = tweets_and_topics["dominant_topic"].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ["Topic Number", "Number of Tweets"]
    print(df_topic_distribution)

    # Display word cloud
    displayWordcloud(ldaModel, tfFeatureNames)

    # Apply sentiment analysis to each constructed topics
    # returns tweetTokens and tweetDates
    # do this again to obtain tweet tokens in a format ready for sentiment analysis
    tweets = tp.getTweetDf(jsonFilename, removeFreqWords=True)

    # get sentiments for each tweet
    vaderSentiments = sa.vaderSentimentAnalysis(tweets["tweetTokens"], printSentiment=False)

    # preprate data for plotting
    tweet_df = pd.DataFrame({"Sentiments": vaderSentiments,
                             "Date": tweets["tweetDates"],
                             "DominantTopic": tweet_dominant_topic})
    tweet_df["Sentiments"] = tweet_df["Sentiments"].apply(pd.to_numeric)

    # distribution of sentiments across Topics
    g = ggplot(aes(x='Sentiments'), data=tweet_df) + \
        geom_histogram() + \
        facet_wrap('DominantTopic', nrow=3) + \
        labs(x="Sentiment Score", y="Frequency")
    print(g)



def display_topics(model, featureNames, numTopWords):
    """
    Prints out the most associated words for each feature.
    Note: method obtained from Lecturer Jeffrey Chan

    @param model: lda model.
    @param featureNames: list of strings, representing the list of features/words.
    @param numTopWords: number of words to print per topic.
    """

    # print out the topic distributions
    for topicId, lTopicDist in enumerate(model.components_):
        print("Topic %d:" % (topicId))
        print(" ".join([featureNames[i] for i in lTopicDist.argsort()[:-numTopWords - 1:-1]]))


def displayWordcloud(model, featureNames):
    """
    Displays the word cloud of the topic distributions, stored in model.
    Note: method obtained from Lecturer Jeffrey Chan

    @param model: lda model.
    @param featureNames: list of strings, representing the list of features/words.
    """

    # this normalises each row/topic to sum to one
    normalisedComponents = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    topicNum = len(model.components_)
    # number of wordclouds for each row
    plotColNum = 2
    # number of wordclouds for each column
    plotRowNum = int(math.ceil(topicNum / plotColNum))

    # plot each wordlcloud
    for topicId, lTopicDist in enumerate(normalisedComponents):
        lWordProb = {featureNames[i] : wordProb for i, wordProb in enumerate(lTopicDist)}
        wordcloud = WordCloud(background_color='white')
        wordcloud.fit_words(frequencies=lWordProb)
        plt.subplot(plotRowNum, plotColNum, topicId+1)
        plt.title('Topic %d:' % (topicId))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
    plt.show(block=True)


if __name__ == '__main__':
    main()

