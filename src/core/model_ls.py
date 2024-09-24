from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .core_utils import CoreUtils
from .text_processor import TextPreprocessor
from .auto_labeling import AutoLabeling
from src.extensions.os_extension import OsOperation

#----------- Model Save/Load Func ----------
def save_model_LS(model, vectorizer, label_encoder, unmapped_encoder, name):
    # Save both the model and vectorizer to a file

    PATH_PREFIX = f'data/model_store/{name}'

    osOperation = OsOperation()
    file_list = osOperation.get_dir_files(PATH_PREFIX)

    next_version = len(file_list) + 1

    dump_name = f'{PATH_PREFIX}/model_LS_{name}_{next_version}.pkl'
    
    # code adapted from Sharma (2023)
    joblib.dump((model, vectorizer, label_encoder, unmapped_encoder), dump_name)
    # end of adapted code
    print(f"{name} Model and vectorizer saved successfully")

def load_model_LS(name, version=0):
    # Load both the model and vectorizer from the file

    PATH_PREFIX = f'data/model_store/{name}'

    if version == 0:
        dump_name = f'{PATH_PREFIX}/model_LS_{name}_1.pkl'
    else:
        dump_name = f'{PATH_PREFIX}/model_LS_{name}_{version}.pkl'
    
    # code adapted from Sharma (2023)
    model, vectorizer,label_encoder, unmapped_encoder = joblib.load(dump_name)
    # end of adapted code
    print("Model and vectorizer loaded successfully")
    return model, vectorizer, label_encoder, unmapped_encoder
#----------- Model Save/Load Func ----------

 #----------- Visualization Func ----------
# Show Label Distribution
def show_label(df, label_col):
    plt.figure(figsize=(8,6))
    sns.countplot(df[label_col])
    plt.title('The distribution of Primary problem')

# confusion_matrix
def show_confusion_matrix(model, X_test, y_test, label_encoder):
    # Predict the classes for the test set
    y_pred = model.predict(X_test)

    label_classes = label_encoder.classes_
    all_labels = label_encoder.fit_transform(label_classes)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
#----------- Visualization Func ----------

class ModelLS:

    def __init__(self, dfs=None, ds_name='asrs', sample_size=0, options={
        "is_merge_taxonomy": False
    }):

        self.ds_name = ds_name
        self.sample_size = sample_size
        self.options = options

        factor_col_name = CoreUtils.get_constant()["FACTOR_COL_NAME"]
        self.factor_col_name = factor_col_name

        if dfs is None:
            print("The DataFrame is None")
            dfs = self.get_data()
        
        self.dfs = dfs

        df_labeled = self.do_label()

        df_labeled = df_labeled.dropna(subset=['HFACS_Category_Value', factor_col_name])

        labels = df_labeled["HFACS_Category_Value"].values
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(labels)

        print('Shape of label tensor:', Y.shape)

        self.df_labeled = df_labeled
        self.label_encoder = label_encoder
        self.Y = Y

    def get_data(self):
        ds_name = self.ds_name
        ds_name_list = ds_name.split('_')

        dfs = {}

        for ds_name_item in ds_name_list:
            df_item = CoreUtils.get_data(ds_name_item)
            dfs[ds_name_item] = df_item
        
        return dfs

    def do_label(self):
        sample_size = self.sample_size
        options = self.options

        ds_name = self.ds_name
        dfs = self.dfs.copy()

        ds_name_list = ds_name.split('_')

        if len(ds_name_list) > 1:
            labeled_dfs = []
            ds_name_item_01 = ds_name_list[0]
            df = dfs[ds_name_item_01]
            
            autoLabeling_01 = AutoLabeling(df, ds_name_item_01, options)
            df_labeled = autoLabeling_01.do_auto_label(sample_size)
            labeled_dfs.append(df_labeled)

            ds_name_list.pop(0)

            for ds_name_item in ds_name_list:
                df_next = dfs[ds_name_item]
                autoLabeling = AutoLabeling(df_next, ds_name_item, options)
                df_labeled_next = autoLabeling.do_auto_label(sample_size)
                labeled_dfs.append(df_labeled_next)
            
            return pd.concat(labeled_dfs, axis=0).reset_index(drop=True)

        df = dfs[ds_name]
        autoLabeling = AutoLabeling(df, ds_name, options)
        df_labeled = autoLabeling.do_auto_label(sample_size)
        return df_labeled

    # Model Train
    def train(self):

        ds_name = self.ds_name
        ls_df = self.df_labeled.copy()
        factor_column_name = self.factor_col_name

        # print(self.Y)

        label_encoder, y = [self.label_encoder, self.Y] 
        unmapped_encoder = label_encoder.transform(['Unmapped'])[0]

        # Set unlabeled data points to -1 (required by LabelSpreading)
        y[y == unmapped_encoder] = -1

        # print(unmapped_encoder, y)

        # Vectorize the 'Combined_Factors' column using TF-IDF
        textPreprocessor = TextPreprocessor()
        vectorizer, X,  = textPreprocessor.tfidf_vectorization(ls_df, factor_column_name) 

        # Split the labeled data into train and test sets (for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(X[y != -1], y[y != -1], test_size=0.2, random_state=42)

        # Create and fit the LabelSpreading model using ALL data (labeled + unlabeled)
        label_prop_model = LabelSpreading()
        label_prop_model.fit(X, y)

        # Make predictions on the test set (only labeled data for evaluation)
        y_test_predict = label_prop_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_predict)
        print("Accuracy on test data:", accuracy)

        # Optionally show confusion matrix
        show_confusion_matrix(label_prop_model, X_test, y_test, label_encoder)

        # Predict the labels for the full dataset (including unlabeled data)
        y_full_predict = label_prop_model.predict(X)

        # Handle -1 predictions by replacing them with 'Unmapped'
        y_full_predict = np.where(y_full_predict == -1, unmapped_encoder, y_full_predict)

        # Decode the predicted labels back to their original categories
        y_decoded = label_encoder.inverse_transform(y_full_predict)

        # Add the predicted labels back to the DataFrame
        ls_df['HFACS_Category_Value_Predict'] = y_decoded

        # Save Model/Vectorizer and label_encoder
        save_model_LS(label_prop_model, vectorizer, label_encoder, unmapped_encoder, ds_name)

    # Model predict function
    @staticmethod
    def predict(df: pd.DataFrame, ds_name='asrs', version=0, sample_size=0):

        factor_col_name = CoreUtils.get_constant()["FACTOR_COL_NAME"]

        model, vectorizer, label_encoder, unmapped_encoder = load_model_LS(ds_name, version)

        # New set of data for prediction
        print("LS sample_size====", sample_size)
        if sample_size > 0:
            labeled_data =  df.tail(sample_size).copy()
        else:
            labeled_data =  df

        print("Factors Null count", labeled_data[factor_col_name].isnull().sum())

        # Drop rows with NaN in 'finding_factors'
        labeled_data = labeled_data.dropna(subset=[factor_col_name])

        # Vectorize the new set of data
        X_vec = vectorizer.transform(labeled_data[factor_col_name])  

        # Predict using the trained LabelSpreading model
        y = model.predict(X_vec)

        # Replace -1 values with 'Unmapped' placeholder
        y = np.where(y == -1, unmapped_encoder, y)

        # Decode the predicted labels
        y_decoded = label_encoder.inverse_transform(y)

        # Add predictions to the new DataFrame
        labeled_data = labeled_data.copy() 
        labeled_data['HFACS_Category_Value_Predict'] = y_decoded

        return labeled_data


    # https://www.analyticsvidhya.com/blog/2023/02/how-to-save-and-load-machine-learning-models-in-python-using-joblib-library/