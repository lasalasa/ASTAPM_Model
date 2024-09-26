
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd

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

import joblib

from src.core.core_utils import CoreUtils
from src.core.text_processor import TextPreprocessor
from src.core.model_ls import ModelLS

# This should be the same as the 'num_words' in the tokenizer
MAX_NB_WORDS = 7500
# This is fixed.
EMBEDDING_DIM = 100

MAX_LENGTH = 300

def show_label_distribution(data):
    plt.figure(figsize=(8,6))
    sns.countplot(data['HFACS_Category_Value_Predict'])
    plt.title('The distribution of Primary problem')

def word_count_distribution(df):
    # Assuming you have a list of all words in your dataset
    all_words = [word for text in df['narrative'].values for word in text.split()]
    word_counts = Counter(all_words)

    # Plot top 50 most frequent words
    common_words = word_counts.most_common(50)
    labels, values = zip(*common_words)
    plt.bar(labels, values)
    plt.xticks(rotation=90)
    plt.show()

    # Print the number of unique words in your dataset
    print(f"Total unique words: {len(word_counts)}")

def show_narrative_distribution(data):
    plt.figure(figsize=(14, 6))

    word_count = data['narrative_word_count']

    sns.histplot(word_count,  bins=50, color='blue', kde=True)
    plt.xlabel('Number of Words')
    plt.ylabel('Number of sample')
    plt.title('Distribution of Word Counts (KDE)')

    # Overlay mean (mu) and standard deviation (sigma) on the plot
    mean = word_count.mean()
    std = word_count.std()
    # Display mean and standard deviation on the plot
    plt.text(0.7, 0.9, r'$\mu={:.2f}$'.format(mean), transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.7, 0.85, r'$\sigma={:.2f}$'.format(std), transform=plt.gca().transAxes, fontsize=12)

    # Display the plots
    plt.tight_layout()
    plt.show()

# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

def model_summary(history):

    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the loss on the first axes
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='test')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot the accuracy on the second axes
    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='test')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    

#  Confusion Matrix
def show_lstm_confusion_matrix(conf_matrix, classes):
    # Plot the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


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
    def __init__(self, dfs=None, ds_name='asrs', ls_version=1, sample_size=1000, max_length=300, max_nb_words=5000, is_enable_smote=False):

        self.ds_name = ds_name
        self.sample_size = sample_size
        self.ls_version = ls_version
        self.max_length = max_length
        self.max_nb_words = max_nb_words
        self.is_enable_smote = is_enable_smote

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

        show_label_distribution(df)
        show_narrative_distribution(df)
        word_count_distribution(df)

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
        
        model = Sequential()
        model.add(Embedding(max_nb_words, EMBEDDING_DIM))
        model.add(SpatialDropout1D(0.25))
        model.add(LSTM(units=64, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))
        model.add(LSTM(units=64, dropout=0.25, recurrent_dropout=0.25))
        model.add(Dense(len(lstm_label_encoder.classes_), activation='softmax', kernel_regularizer=l2(0.001)))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),  metrics=['accuracy'])

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

    def define_hyperparameters(self, options):
        # Hyperparameters 
        self.max_nb_words = 10000 # max number of words to use in the vocabulary
        self.max_length = 100 # max length of each text (in terms of number of words)
        self.embedding_dim = 100 # dimension of word embeddings
        self.lstm_units = 64 # number of units in the LSTM layer 
        self.num_of_class = len(self.lstm_label_encoder.classes_)
    
    def define_model(self):
        model = Sequential()
        model.add(Embedding(self.max_nb_words, self.embedding_dim))
        model.add(SpatialDropout1D(0.25))
        model.add(LSTM(units=self.lstm_units, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))
        model.add(LSTM(units=64, dropout=0.25, recurrent_dropout=0.25))
        model.add(Dense(len(self.num_of_class), activation='softmax', kernel_regularizer=l2(0.001)))

    def pre_process(self, dfs):
        ds_name = self.ds_name
        sample_size = self.sample_size
        ls_version = self.ls_version
        factor_col_name = self.factor_col_name

        df = dfs[ds_name]

        # 02. Label Spreading
        print("Start labelling")
        df = ModelLS.predict(df, ds_name, ls_version, sample_size)
        print("Ladled Sampling size=", df.shape)

        print("start pre_process_df")
        df = pre_process_df(df, factor_col_name)

        dfs[ds_name] = df

        # print(df.head())
        return dfs


    def train(self, epochs=10, batch_size=32):

        X = self.X
        Y = self.Y
        is_enable_smote = self.is_enable_smote

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)
        
        print(X_train.shape,Y_train.shape)
        print(X_test.shape,Y_test.shape)

        # print(pd.Series(Y_train).value_counts())
        print(pd.Series(Y_test).value_counts())
        
        if is_enable_smote:

            smote = SMOTE(random_state=42)
            X_resampled, Y_resampled = smote.fit_resample(X_train, Y_train)

            print(f"Original dataset shape: {X_train.shape}")
            print(f"Resampled dataset shape: {X_resampled.shape}")

        else: 
            X_resampled = X_train
            Y_resampled = Y_train


        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', 
            patience=2, 
            min_delta=0.001)
        
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

        model_summary(history)
        show_lstm_confusion_matrix(conf_matrix, labels)

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