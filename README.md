# Reddit Sentiment Analysis Project
This project involves mining data from Reddit, performing sentiment analysis on user comments, and applying machine learning models to classify these sentiments. The project is divided into two parts: data collection and sentiment classification.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Program 1: Reddit Data Mining](#program-1-reddit-data-mining)
  - [Overview](#overview)
  - [Subreddits and Keywords](#subreddits-and-keywords)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Data Saving](#data-saving)
- [Program 2: Sentiment Classification and Modeling](#program-2-sentiment-classification-and-modeling)
  - [Overview](#overview-1)
  - [Preprocessing](#preprocessing)
  - [Modeling and Evaluation](#modeling-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)

## Introduction
This project uses the Python Reddit API Wrapper (PRAW) to collect user comments from various Tesla-related subreddits. These comments are analyzed for sentiment using the TextBlob library and then processed to build a machine learning model that can classify sentiments into positive, neutral, and negative categories.

- The project consists of two main parts:

Reddit Data Mining: Collecting comments related to specific Tesla models.
Sentiment Classification: Analyzing and classifying the collected comments based on sentiment.
## Project Structure
- reddit_data_mining.py: This script collects data from Reddit and performs sentiment analysis.
- sentiment_classification.py: This script preprocesses the collected data, applies machine learning models, and evaluates their performance.
## Installation
Prerequisites
Before running the scripts, ensure you have the following installed:

```Python 3.x
PRAW
TextBlob
Pandas
Numpy
Matplotlib
Seaborn
Scikit-learn
NLTK
```
To install the required libraries, run:

```bash
pip install praw textblob pandas numpy matplotlib seaborn scikit-learn nltk
```
## Program 1: Reddit Data Mining
### Overview
The first part of the project involves mining data from multiple Tesla-related subreddits using the PRAW library. The script collects comments related to the Tesla Model Y and analyzes the sentiment of each comment using TextBlob.

### Subreddits and Keywords
The script collects data from the following subreddits:

- TeslaModelS
- teslamotors
- TeslaLounge
- SpaceXMasterrace
- RealTesla
Relevant keywords related to Tesla Model Y are used to search for posts in the subreddits, such as:

- "Model Y"
- "Model Y Performance"
- "Model Y Review"
- "Model Y Durability"
- "Model Y Safety"
### Sentiment Analysis
For each comment retrieved from the posts, TextBlob is used to perform sentiment analysis. The sentiment is classified into three categories:

- Positive: If the polarity score is greater than 0.
- Neutral: If the polarity score is 0.
- Negative: If the polarity score is less than 0.
The sentiment is saved in a DataFrame along with the comment text and the upvote score.

### Data Saving
The collected data, including the comments, upvotes, and sentiment scores, are saved into a CSV file reddit_customer_comments1.csv. The file is then downloaded to the local machine.

## Program 2: Sentiment Classification and Modeling
### Overview
The second part of the project focuses on further processing the collected data and applying machine learning techniques to classify sentiments more accurately.

### Preprocessing
- Data Cleaning: The sentiment column from the previous analysis is removed due to incorrect results.
- Sentiment Re-analysis: The script applies TextBlob once again to correctly label the sentiment for each comment.
- Data Balancing: The dataset is highly imbalanced, so undersampling is performed to balance the number of comments for each sentiment category (positive, neutral, negative).
### Feature Engineering
Text Preprocessing: Each comment is preprocessed by:
- Lowercasing
- Removing URLs and special characters
- Tokenization
- Removing stopwords
- Stemming
### Modeling and Evaluation
- TF-IDF Vectorization: A TfidfVectorizer is used to convert the preprocessed text data into numerical features.
- Model Training: A Decision Tree Classifier is trained on the data to predict the sentiment.
- Model Evaluation: The trained model is evaluated using the following metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
The evaluation results are printed, and a bar chart of the sentiment distribution is displayed before and after undersampling to show the impact of balancing the dataset.

## Results
The first part of the project successfully collects data from Reddit, with comments related to the Tesla Model Y.
Sentiment analysis reveals that most comments are positive, but after balancing the data, the classifier is able to make more accurate predictions across all sentiment classes.
The machine learning model demonstrates satisfactory performance in classifying sentiment, but improvements could be made with more data and advanced modeling techniques.
## Conclusion
This project demonstrates how data from Reddit can be collected and analyzed for sentiment using both basic and machine learning techniques. While the initial analysis shows that positive sentiments are dominant, rebalancing the dataset allows for a more comprehensive sentiment classification model. Future improvements could involve using more sophisticated models like random forests, support vector machines, or deep learning for better results.

## Dependencies

To run the project, ensure you have the following dependencies installed:

- **[praw](https://praw.readthedocs.io/):** For Reddit API access
- **[textblob](https://textblob.readthedocs.io/):** For sentiment analysis
- **[pandas](https://pandas.pydata.org/):** For data manipulation
- **[numpy](https://numpy.org/):** For numerical operations
- **[matplotlib](https://matplotlib.org/):** For data visualization
- **[seaborn](https://seaborn.pydata.org/):** For enhanced visualizations
- **[scikit-learn](https://scikit-learn.org/):** For machine learning models
- **[nltk](https://www.nltk.org/):** For natural language processing tasks

You can install the dependencies using the following command:
```bash
pip install praw textblob pandas numpy matplotlib seaborn scikit-learn nltk
```

