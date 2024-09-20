import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
import numpy as np

import pandas as pd
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict

class DataPreprocessor:
    def __init__(self, technique='NMF', n_components=5):
        self.technique = technique
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.feature_engineer = None
    
    def preprocess(self, data: pd.DataFrame):

        data['narrative'] = data['narrative_01'] + data['narrative_02']

        data['narrative'] = [entry.lower() for entry in data['narrative']]

        # Handle missing values
        data.fillna(method='ffill', inplace=True)
        
        # Standardize features
        scaled_data = self.scaler.fit_transform(data)
        
        # Feature engineering based on the selected technique
        if self.technique == 'NMF':
            self.feature_engineer = NMF(n_components=self.n_components)
            transformed_data = self.feature_engineer.fit_transform(scaled_data)
        else:
            raise ValueError(f"Unsupported feature engineering technique: {self.technique}")
        
        return transformed_data

    def clean_and_tokenize_narratives(self, data: pd.DataFrame):

        # Step - a : Remove blank rows if any in the 'narrative' columns.
        data['narrative'] = data['narrative_01'] + ' ' + data['narrative_02']
        data.dropna(subset=['narrative'], inplace=True)

        # Step - b : Change all the text to lower case.
        data['narrative'] = data['narrative'].str.lower()

        # Step - c : Tokenization : In this each entry in the corpus will be broken into a set of words
        data['narrative'] = data['narrative'].apply(word_tokenize)

        # Step - d : Remove Stop words, Non-Numeric and perform Word Stemming/Lemmatizing
        # Initialize the tag map for lemmatization
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        # Initialize WordNetLemmatizer
        word_Lemmatized = WordNetLemmatizer()

        # Function to process text
        def process_text(entry):
            final_words = []
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                    final_words.append(word_final)
            return final_words

        # Apply the text processing to each entry in 'narrative'
        data['narrative_pre_processed'] = data['narrative'].apply(process_text)

        # Optional: Convert the list of words back to a string (if needed)
        data['narrative_pre_processed'] = data['narrative_pre_processed'].apply(lambda x: ' '.join(x))

        return data

# https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions