from flask import Flask, request, render_template
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# for text pre-processing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
# Polarity Scores
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')


# Declare a Flask app
app = Flask(__name__)

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}


# The code for expanding contraction words
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # Tokenizing text into tokens.
    list_of_tokens = text.split(' ')

    # Checking for whether the given token matches with the Key & replacing word with key's value.

    # Check whether Word is in list_Of_tokens or not.
    for Word in list_of_tokens:
        # Check whether found word is in dictionary "Contraction Map" or not as a key.
        if Word in CONTRACTION_MAP:
            # If Word is present in both dictionary & list_Of_tokens, replace that word with the key value.
            list_of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_of_tokens]

    # Converting list of tokens to String.
    string_of_tokens = ' '.join(str(e) for e in list_of_tokens)
    return string_of_tokens


lemmatizer = WordNetLemmatizer()


def lemmatize_sentence(sentence):
    token_words=word_tokenize(sentence)
# we need to tokenize the sentence or else lemmatizing will return the entire sentence as is.
    lemma_sentence=[]
    for word in token_words:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(' ')
    return ''.join(lemma_sentence)


stop_list = nltk.corpus.stopwords.words("english")
stop_list = set(stop_list)


def remove_stopwords(text):
    # repr() function actually gives the precise information about the string
    text = repr(text)
    # Text without stopwords
    so_stopWords = [word for word in word_tokenize(text) if word.lower() not in stop_list]
    # Convert list of tokens_without_stopwords to String type.
    words_string = ' '.join(so_stopWords)
    return words_string


def pre_process(tweet):
    # lowe Case
    lowercase_tweet = tweet.str.lower()
    # expand contractions
    tweet_expanded = lowercase_tweet.apply(lambda x: expand_contractions(x))

    # remove extra spaces
    tweet_space = tweet_expanded.str.replace(r'\s+', ' ')

    # remove @user
    tweet_no_user = tweet_space.str.replace(r'@[\w\-]+', ' ')

    # remove links
    url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    tweet_no_links = tweet_no_user.str.replace(url_regex, ' ')

    # remove stopwords
    tweet_stopwords_removed = tweet_no_links.apply(lambda x: remove_stopwords(x))

    # remove punctuation and numbers
    tweet_punc_nums = tweet_stopwords_removed.str.replace('[^a-zA-Z]', ' ')

    # remove whitespace
    tweet = tweet_punc_nums.str.replace(r'\s+', ' ')
    # remove leading and trailing space
    tweet = tweet.str.replace(r'^\s+|\s+?$', ' ')

    # tokenizing and lemmenization
    lemmatized_tweet = tweet.apply(lambda x: lemmatize_sentence(x))

    for i in range(len(lemmatized_tweet)):
        lemmatized_tweet[i] = ''.join(lemmatized_tweet[i])
        tweets_p = lemmatized_tweet

    return tweets_p


sentiment_analyzer = VS()


def count_tags(tweet_c):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', tweet_c)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def sentiment_analysis(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    twitter_objs = count_tags(tweet)
    features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'], twitter_objs[0],
                twitter_objs[1], twitter_objs[2]]
    # features = pandas.DataFrame(features)
    return features


def sentiment_analysis_array(tweets):
    features = []
    for t in tweets:
        features.append(sentiment_analysis(t))
    return np.array(features)


@app.route('/', methods=['GET','POST'])
def main():

    if request.method == "POST":

        # Unpickle classifier
        model = pickle.load(open("rf_class.pkl", 'rb'))
        vectorizer = pickle.load(open("tfidf.pkl", 'rb'))

        # Get values through input bars
        text = request.form.get("text_input")

        # Put inputs to dataframe
        X = {'message': [text]}

        X_df = pd.DataFrame(X)

        X_text = X_df.message

        X_preprocessed = pre_process(X_text)

        X_vecorised = vectorizer.transform(X_preprocessed)

        X_final = sentiment_analysis_array(X_text)

        input = X_vecorised.toarray()

        X_features = np.concatenate([input, X_final], axis=1)

        final_X = pd.DataFrame(X_features)

        final_X

        # Get prediction
        prediction = model.predict(final_X)

        if prediction == 0:
            prediction = "Not Hate Speech"
        else: prediction = "Hate Speech"

    else:
        prediction = ""

    return render_template("demo.html", output=prediction)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
