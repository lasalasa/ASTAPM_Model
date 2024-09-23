# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from gensim.models import Word2Vec

# nltk.download('punkt')
# nltk.download('stopwords')

#library that contains punctuation
import string
string.punctuation
import pandas as pd

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#importing nlp library
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

#importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, acronyms=None):
        self.porter_stemmer = PorterStemmer()
        #defining the object for Lemmatization
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def combined_narrative(self, df):
        df['narrative_02'] = df['narrative_02'].fillna('')

        # Step 4: Combine 'narrative_01' and 'narrative_02' into a single 'narrative' column
        # code adapted from GeeksforGeeks (2021)
        df['narrative'] = df['narrative_01'].str.cat(df['narrative_02'], sep=' ', na_rep='')
        # end of adapted code
        return df

    def counting_narrative(self, df):
        df['narrative_length'] = df['narrative'].apply(len)
        # Count words in each narrative
        df['narrative_word_count'] = df['narrative'].apply(lambda x: len(x.split()))

        # Count sentences in each narrative
        df['narrative_sentence_count'] = df['narrative'].apply(lambda x: len(x.split('. ')))
        
        return df

    # def clean_narrative(self, df):
    #     df.dropna(subset=['narrative'], inplace=True)

    #     df = df[df['narrative_word_count'] > 0].copy()
    #     return df

    def clean_feature(self, df: pd.DataFrame, feature_name):
        df = df.dropna(subset=[feature_name])

        return df

    def drop_narratives(self, df):
        # Drop column narrative 01 and 02
        df = df.drop('narrative_01', axis=1)
        df = df.drop('narrative_02', axis=1)
        return df

    def show_summary(self, df: pd.DataFrame):
        null_summary = df.isnull().sum()
        print(null_summary)
        print(df.head())

    ## -------------- NLP Preprocessing

    # code adapted from Deepanshi (2024)
    # https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/#:~:text=Text%20preprocessing%20is%20an%20essential,part%2Dof%2Dspeech%20tagging.
    #defining the function to remove punctuation
    def remove_punctuation(self, text):
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree

    def do_lower(self, df, col_name):
        df[col_name]= df[col_name].apply(lambda x: x.lower())
        return df

    # code adapted from (NLTK :: Nltk.Tokenize Package, n.d.)
    # https://www.nltk.org/api/nltk.tokenize.html
    def tokenization(self, text):
        return word_tokenize(text)
    # end of adapted code

    def remove_stopwords(self, text):
        output = [word for word in text if word not in stop_words]
        return output

    #defining a function for stemming
    def stemming(self, text):
        #defining the object for stemming
        stem_text = [self.porter_stemmer.stem(word) for word in text]
        return stem_text

    #defining the function for lemmatization
    def lemmatizer(self, text):
        lemm_text = [self.wordnet_lemmatizer.lemmatize(word) for word in text]
        return lemm_text
    # end of adapted code
    
    def preprocess_narrative(self, df):
        col_name = 'narrative' #'narrative_cleaned'
        df[col_name]= df[col_name].apply(lambda x: self.remove_punctuation(x))

        df = self.do_lower(df, col_name)

        df[col_name] = df[col_name].apply(lambda x: self.tokenization(x))

        df[col_name] = df[col_name].apply(lambda x: self.remove_stopwords(x))

        df[col_name] = df[col_name].apply(lambda x: self.stemming(x))

        df[col_name] = df[col_name].apply(lambda x: self.lemmatizer(x))

        df[col_name] = df[col_name].apply(lambda x: ' '.join(x))

        return df

    ## -----------------

    # https://www.analyticsvidhya.com/blog/2021/07/bag-of-words-vs-tfidf-vectorization-a-hands-on-tutorial/
    # https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/#:~:text=Bag%20of%20Words%20just%20creates,less%20important%20ones%20as%20well.

    # code adapted from Mimi (2021)
    # Bag-of-words using Count Vectorization
    @staticmethod
    def count_vectorization(df, col_name):
        corpus =df[col_name]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        print(vectorizer.get_feature_names())
        return vectorizer, X

    # TFIDF Vectorization
    @staticmethod
    def tfidf_vectorization(df, col_name):
        corpus =df[col_name]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        return vectorizer, X
    # end of adapted code
