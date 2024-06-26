# -*- coding: utf-8 -*-
"""cust_datamining.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vn3OK-La5d_1WhfP-XDgt_jn8EFxn6pG
"""

''' This code is just foir data mining purposes from reddit
PRAW, an acronym for “Python Reddit API Wrapper”,
PRAW is a python package that allows for simple access to reddit’s API.
PRAW aims to be as easy to use as possible and is designed to follow all of reddit’s API rules. '''
!pip install praw
import praw
from textblob import TextBlob # used for sentiment analysis
import numpy as np
import pandas as pd
from google.colab import files # for downloading purposes
# I need an instance of reddit class to do anything in praw so generating an instance below
reddit = praw.Reddit(
    client_id="vid8c43do99jU9EiWUZsDg",
    client_secret="secretid",
    user_agent="YourAppBot"
)

# now i need subreddits from which i am going to obtain data
# i am saving these subreddits in the variable subreddit
subreddits = list(set(["TeslaModelS", "teslamotors", "TeslaLounge", "SpaceXMasterrace", "teslamotors", "RealTesla"]))

# Adding the relevant keywords as a list
keywords = ["Model Y", "Tesla Model Y", "Model Y 2023", "Model Y Long Range", "Model Y Performance",
            "Model Y Refresh", "Model Y Review", "Model Y Feedback", "Model Y Opinion", "Model Y Experience",
            "Model Y Thoughts", "Model Y Impressions", "Model Y Comments", "Model Y Reactions",
            "Model Y Reliability", "Model Y Durability", "Model Y Safety", "Model Y Comfort", "Model Y Design",
            "Model Y Features", "Model Y Like", "Model Y Love", "Model Y Great", "Model Y Dislike", "Model Y Hate",
            "Model Y Problem", "Model Y Issue", "Model Y Concern", "Model Y Complaint", "Model Y Praise",
            "Model Y Criticism", "Model Y Autonomous Driving", "Model Y Battery Range", "Model Y Charging Infrastructure",
            "Model Y Interior Design", "Model Y Infotainment System", "Model Y Sustainability", "Model Y Price"]
limit = 10

# Initialize lists to store comment data
comment_texts = []
comment_scores = []
comment_sentiments = []

# Defining search parameters
time_filter = "week"  # Search posts from the last week

# Collect comments from each subreddit
for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)

    # Search for relevant posts
    for keyword in keywords:
        search_query = f"{keyword} timestamp:{time_filter}"
        relevant_posts = subreddit.search(search_query, time_filter=time_filter, limit=limit)

        # Process each relevant post
        for post in relevant_posts:

            # Collect comments from the post
            post.comments.replace_more(limit=None)
            for comment in post.comments.list():
                comment_texts.append(comment.body)
                comment_scores.append(comment.score)
                 # Analyze sentiment of the comment using TextBlob
                blob = TextBlob(comment.body)
                sentiment = blob.sentiment.polarity

                # Map sentiment to -1, 0, or 1
                if sentiment < 0:
                  sentiment = -1
                elif sentiment == 0:
                  sentiment = 0
                else:
                  sentiment = 1
                comment_sentiments.append(sentiment)

data={
    "comment":comment_texts,
    "upvotes":comment_scores,
    "sentiment":sentiment
}
df=pd.DataFrame(data)
df

# DataFrame is named df and  saving it to a file named "reddit_comments.csv"
df.to_csv("reddit_customer_comments1.csv", index=False, header=True)

# Download the file to your local machine
files.download("reddit_customer_comments1.csv")
