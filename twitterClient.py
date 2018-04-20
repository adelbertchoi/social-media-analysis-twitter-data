
"""
Assignment1: Code Part 1
by Adelbert Choi

This script aims to retrieve twitter data using REST search API
"""

import sys
import tweepy as tw
import json

from tweepy import Cursor

def twitterAuth():
    """
        Setup Twitter API authentication.
        Note: method obtained from Lecturer Jeffrey Chan

        @returns: tweepy.OAuthHandler object
    """
    try:
        consumerKey = ""
        consumerSecret = ""
        accessToken = ""
        accessSecret = ""
    except KeyError:
        sys.stderr.write("Key or secret token are invalid.\n")
        sys.exit(1)

    auth = tw.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessToken, accessSecret)

    return auth



def twitterClient():
    """
        Setup Twitter API client.
        Note: method obtained from Lecturer Jeffrey Chan

        @returns: tweepy.API object
    """
    auth = twitterAuth()
    # wait_on_rate_limit - allows a rest when twitter limit is reached, and continue once permitted
    # wait_on_rate_limit_notify - notifies when twitter limit is reached
    client = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    return client


def main():
    client = twitterClient()

    # obtain location id for USA
    # places = client.geo_search(query="USA", granularity="country")
    # place_id = places[0].id
    # print('USA id is: ', place_id)

    # obtain obtain tweets with keywords uber, #uber from the US
    # also tweets classified as retweets are filtered out
    searchQuery = "place:96683cc9126741d1 uber OR #uber -filter:retweets"
    # maximum number of tweets to retrieve
    maxTweets = 1000000
    # tweet counter
    tweetCount = 0

    # Rest Search API to obtain tweets based on searchQuery, also, only return english tweets lang="en"
    uber = Cursor(client.search, q=searchQuery, since="2018-03-22", until="2018-04-02", lang="en").items(maxTweets)

    # save and add retrieved tweets to a json file
    with open("uberTweetsUS.json", "w") as fOut:
        for tweet in uber:
            fOut.write("{}\n".format(json.dumps(tweet._json)))
            print(tweetCount+1, " ", tweet.created_at, " ", tweet.text) # print out each tweet collected
            tweetCount += 1

        # Display how many tweets are collected
        print("Downloaded {0} tweets".format(tweetCount))

    # save subset of data to another json file for submission
    # save the first 100 tweets in the data retrieved to a new json file
    count=0
    with open("uberTweetsUS_subset.json", "w") as fOut:
        with open("uberTweetsUS.json", 'r') as file:
            for line in file:
                if count != 100:
                    tweet = json.loads(line)
                    fOut.write("{}\n".format(json.dumps(tweet)))
                    count+=1
                    print(count)
                if count == 100:
                    break



if __name__ == "__main__":
    main()
