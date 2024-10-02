from sklearn.preprocessing import LabelEncoder

# Text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.utils import class_weight
import numpy as np


import numpy as np
import pandas as pd
from collections import Counter

import joblib

from src.core.core_utils import CoreUtils
from src.core.text_processor import TextPreprocessor
from src.core.model_ls import ModelLS
from src.core.model_lstm_plot import word_count_distribution, show_label_distribution, show_lstm_confusion_matrix, show_narrative_distribution, model_summary

# This should be the same as the 'num_words' in the tokenizer
MAX_NB_WORDS = 7500
# This is fixed.
EMBEDDING_DIM = 100

MAX_LENGTH = 300

def pre_process_df(df, col_name):

    textPreprocessor = TextPreprocessor()
     # Step 1: Combine 'narrative_01' and 'narrative_02' into a single 'narrative' column
    print("combined_narrative")
    df = textPreprocessor.combined_narrative(df)
    
    # print("counting_narrative")
    # # Step 2: Combine 'narrative_01' and 'narrative_02' into a single 'narrative' column
    # df = textPreprocessor.counting_narrative(df)

    # print("clean_narrative")
    # # Step 3: clean narrative
    # df = textPreprocessor.clean_narrative(df)

    print("clean_feature")
    # Step 4: Clean factor_column_name
    df = textPreprocessor.clean_feature(df, col_name)

    print("drop_narratives")
    # Step 5: Drop column narrative 01 and 02
    df = textPreprocessor.drop_narratives(df)

    print("preprocess_narrative")
    df = textPreprocessor.preprocess_narrative(df)
    df = textPreprocessor.counting_narrative(df)

    # Step 6: Show summary
    textPreprocessor.show_summary(df)

    return df

class LSTMModel:
    def __init__(self, dfs=None, ds_name='asrs', options={
        "ls_name": 'asrs_ntsb',
        "ls_version": 1,
        "sample_size":1000,
        "max_length":300, 
        "max_nb_words":5000, 
        "is_enable_smote":False,
        "is_enable_asasyn": False,
        "is_enable_class_weight": False,
    }):
        self.ds_name = ds_name
        self.options = options
        self.sample_size = options['sample_size']

        max_length = options['max_length']
        self.max_length = max_length

        max_nb_words = options['max_nb_words']
        self.max_nb_words  = max_nb_words 
        self.is_enable_smote = options['is_enable_smote']
        self.is_enable_asasyn = options['is_enable_asasyn']
        self.is_enable_class_weight = options['is_enable_class_weight']

        factor_col_name = CoreUtils.get_constant()["LS_CLASSIFICATION_FACTOR"]
        self.factor_col_name = factor_col_name

        if dfs is None:
            print("The DataFrame is None")
            dfs = self.get_data()        
            print("Data loaded")
        
        # print(dfs[ds_name])

        dfs = self.pre_process(dfs)
        self.dfs = dfs
        print("Pre processed")

        df = dfs[ds_name]

        show_label_distribution(df, ds_name)
        show_narrative_distribution(df, ds_name)
        word_count_distribution(df, ds_name)

        print("Define Y")
        lstm_labels = df['HFACS_Category_Value_Predict'].values
        lstm_label_encoder = LabelEncoder()
        Y = lstm_label_encoder.fit_transform(lstm_labels)
        self.Y = Y
        self.label_encoder = lstm_label_encoder
        print('Shape of label tensor:', Y.shape)

        print("Define X")
        texts = df['narrative'].values
        tokenizer = Tokenizer(num_words=max_nb_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        sequences = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=max_length)
        self.X = X

        print('Shape of data tensor:', X.shape)
        
        # Code adapted from Li (2021)
        model = Sequential()
        model.add(Embedding(max_nb_words, EMBEDDING_DIM))
        model.add(SpatialDropout1D(0.3))
        model.add(LSTM(units=128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
        model.add(LSTM(units=128, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(len(lstm_label_encoder.classes_), activation='softmax', kernel_regularizer=l2(0.001)))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001) ,  metrics=['accuracy'])
        # Code adapted end

        self.model = model

    def get_data(self):
        ds_name = self.ds_name
        ds_name_list = ds_name.split('_')

        # print(ds_name_list)
        self.sample_size = self.sample_size * len(ds_name_list)

        dfs = {}
        dfs_list = []
        for ds_name_item in ds_name_list:
            df_item = CoreUtils.get_data(ds_name_item)
            dfs[ds_name_item] = df_item
            dfs_list.append(df_item)

        dfs[ds_name] = pd.concat(dfs_list, axis=0).reset_index(drop=True)

        return dfs

    def get_hyperparameters(self, options):
        # Hyperparameters 
        self.max_nb_words = 10000 # max number of words to use in the vocabulary
        self.max_length = 100 # max length of each text (in terms of number of words)
        self.embedding_dim = 100 # dimension of word embeddings
        self.lstm_units = 64 # number of units in the LSTM layer 
        self.num_of_class = len(self.lstm_label_encoder.classes_)
    
    def make_model(self):
        model = Sequential()
        model.add(Embedding(self.max_nb_words, self.embedding_dim))
        model.add(SpatialDropout1D(0.3))
        model.add(LSTM(units=self.lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))
        model.add(LSTM(units=self.lstm_units, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(len(self.num_of_class), activation='softmax', kernel_regularizer=l2(0.001)))

    def pre_process(self, dfs):
        ds_name = self.ds_name
        sample_size = self.sample_size
        factor_col_name = self.factor_col_name
        options = self.options

        ls_name = options['ls_name']
        ls_version = options['ls_version']

        df = dfs[ds_name]

        # 02. Label Spreading
        print("Start labelling")
        df = ModelLS.predict(df, ls_name, ls_version, sample_size)
        print("Ladled Sampling size=", df.shape)

        print("start pre_process_df")
        df = pre_process_df(df, factor_col_name)

        dfs[ds_name] = df

        # print(df.head())
        return dfs


    def train(self, epochs=10, batch_size=32, class_weights_dict=None):

        X = self.X
        Y = self.Y
        is_enable_smote = self.is_enable_smote
        is_enable_asasyn =  self.is_enable_asasyn
        is_enable_class_weight = self.is_enable_class_weight

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
        
        print(X_train.shape,Y_train.shape)
        print(X_test.shape,Y_test.shape)

        print(Counter(Y_train))

        if is_enable_smote:

            smote = SMOTE(random_state=42)
            X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

            print(f"Original dataset shape: {X_train.shape}")
            print(f"Resampled dataset shape: {X_resampled.shape}")

        elif is_enable_asasyn:

            ada = ADASYN(sampling_strategy='minority', random_state=130)
            X_resampled, Y_resampled = ada.fit_resample(X_train, Y_train)

            print(f"Original dataset shape: {X_train.shape}")
            print(f"Resampled dataset shape: {X_resampled.shape}")

        else: 
            X_resampled = X_train
            Y_resampled = Y_train

         # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', 
            patience=3, 
            min_delta=0.001)
        
        # print(pd.Series(Y_train).value_counts())
        print(Counter(Y_resampled))
        # print(pd.Series(Y_resampled).value_counts())

        if is_enable_class_weight:

            if class_weights_dict is None:

                class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(Y_resampled), y=Y_resampled)
                class_weights_dict = dict(enumerate(class_weights))

            print(class_weights_dict)

            history = self.model.fit(X_resampled, Y_resampled, 
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weights_dict,
                validation_split=0.1, 
                callbacks=[early_stopping])
        else:
             # Train the model
            history = self.model.fit(X_resampled, Y_resampled, 
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1, 
                callbacks=[early_stopping])
        
        self.history = history
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.save()


    def evaluate(self):
        # Predict the probabilities for the test set
        model = self.model
        ds_name = self.ds_name
        X = self.X
        Y_test = self.Y_test
        X_test = self.X_test
        labels = self.label_encoder.classes_

        map_labels = self.label_encoder.fit_transform(labels)

        y_pred = model.predict(X_test)

        # Convert probabilities to class indices
        y_pred_classes = np.argmax(y_pred, axis=1)
        self.Y_pred = y_pred_classes

        conf_matrix = confusion_matrix(Y_test, y_pred_classes, labels=map_labels)

        # Accuracy
        accuracy = accuracy_score(Y_test, y_pred_classes)
        print(f'Accuracy: {accuracy:.4f}')
        
        report = classification_report(Y_test, y_pred_classes, labels=map_labels, output_dict=True)

        self.classification_report = classification_report(Y_test, y_pred_classes, labels=map_labels)

        # Classification report for precision, recall, F1-score
        # print(classification_report(Y_test, y_pred_classes, labels=map_labels))

        # print(conf_matrix)

        history = self.history

        train_loss = history.history['loss']
        test_loss = history.history['val_loss']

        train_accuracy = history.history['accuracy']
        test_accuracy = history.history['val_accuracy']

        # print(train_loss, test_loss)

        df = self.dfs[ds_name]

        train_data_label_value_count = df['HFACS_Category_Value_Predict'].value_counts().reset_index()

        # Convert DataFrame to dictionary or list of records
        train_data_label_value_count_dict = train_data_label_value_count.to_dict(orient='records')

        # https://docs.python.org/3/library/collections.html#collections.Counter
        all_words = [word for text in df['narrative'].values for word in text.split()]
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(50)
        word_count_labels, word_count_values = zip(*common_words)

        narrative_word_count = df['narrative_word_count'].tolist()

        predict_text = model.predict(X)
        # Convert probabilities to class indices
        predict_text_classes = np.argmax(predict_text, axis=1)

        predict_text_label = self.label_encoder.inverse_transform(predict_text_classes)

        df['lstm_predict_label'] = predict_text_label

        # print(df['date'])
        
        new_df = df.groupby('date')['lstm_predict_label'].value_counts().unstack().fillna(0)

        model_summary(history, ds_name)
        show_lstm_confusion_matrix(conf_matrix, labels, ds_name)

        evaluation_result = {
            "accuracy": accuracy,
            "classification_report": report,
            "conf_matrix": conf_matrix.tolist(),
            "labels": labels.tolist(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_label_value_count": train_data_label_value_count_dict,
            "train_word_count": {
                "labels": word_count_labels,
                "values": word_count_values
            },
            "train_narrative_word_count": narrative_word_count,
            "sample_predict_view": new_df.to_dict(orient="index")
        }

        # print(evaluation_result)

        return evaluation_result

    def predict(self, text):
        loaded_model = self.load()
        predict_text = loaded_model.predict(text)
        predict_text_classes = np.argmax(predict_text, axis=1)
        return predict_text_classes
    
    def save(self):
        ds_name = self.ds_name
        self.model.save(f'{ds_name}.keras')
    
    def load(self):
        ds_name = self.ds_name
        loaded_model = load_model(f'{ds_name}.keras')
        loaded_model.summary()
        return loaded_model

# https://www.kaggle.com/code/khotijahs1/using-lstm-for-nlp-text-classification
# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17