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

#----------- Model Save/Load Func ----------
    # https://www.analyticsvidhya.com/blog/2023/02/how-to-save-and-load-machine-learning-models-in-python-using-joblib-library/
def save_model_LS(model, vectorizer, label_encoder, unmapped_encoder, name):
    # Save both the model and vectorizer to a file
    dump_name = f'model_LS_{name}.pkl'
    joblib.dump((model, vectorizer, label_encoder, unmapped_encoder), dump_name)
    print(f"{name} Model and vectorizer saved successfully")

def load_model_LS(name):
    # Load both the model and vectorizer from the file
    dump_name = f'model_LS_{name}.pkl'
    model, vectorizer,label_encoder, unmapped_encoder = joblib.load(dump_name)
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

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
#----------- Visualization Func ----------

class ModelLS:

    def __init__(self, dfs=None, ds_name='asrs', sample_size=0):

        self.ds_name = ds_name
        self.sample_size = sample_size

        factor_col_name = CoreUtils.get_constant()["FACTOR_COL_NAME"]
        self.factor_col_name = factor_col_name

        if dfs is None:
            print("The DataFrame is None")
            dfs = self.get_data()
        
        self.dfs = dfs

        # autoLabeling = AutoLabeling(df.copy(), ds_name)
        df_labeled = self.do_label()
        #autoLabeling.do_auto_label(5000)

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
        ds_name = self.ds_name
        dfs = self.dfs.copy()

        ds_name_list = ds_name.split('_')

        if len(ds_name_list) > 1:
            labeled_dfs = []
            ds_name_item_01 = ds_name_list[0]
            df = dfs[ds_name_item_01]
            
            autoLabeling_01 = AutoLabeling(df, ds_name_item_01)
            df_labeled = autoLabeling_01.do_auto_label(sample_size)
            labeled_dfs.append(df_labeled)

            ds_name_list.pop(0)

            for ds_name_item in ds_name_list:
                df_next = dfs[ds_name_item]
                autoLabeling = AutoLabeling(df_next, ds_name_item)
                df_labeled_next = autoLabeling.do_auto_label(sample_size)
                labeled_dfs.append(df_labeled_next)

                # df_labeled = pd.concat([df_labeled, df_labeled_next], axis=0).reset_index(drop=True)
            
            return pd.concat(labeled_dfs, axis=0).reset_index(drop=True)

        df = dfs[ds_name]
        autoLabeling = AutoLabeling(df, ds_name)
        df_labeled = autoLabeling.do_auto_label(sample_size)
        return df_labeled

    #----------- Model Train/Predict Func---------
    def train(self):

        ds_name = self.ds_name
        ls_df = self.df_labeled.copy()
        factor_column_name = self.factor_col_name

        label_encoder, y = [self.label_encoder, self.Y] #get_Y_encoder(ls_df, 'HFACS_Category_Value')
        unmapped_encoder = label_encoder.transform(['Unmapped'])[0]
        
        # Set unlabeled data points to -1 (required by LabelSpreading)
        y[y == unmapped_encoder] = -1

        # Step 2: Vectorize the 'Combined_Factors' column using TF-IDF
        textPreprocessor = TextPreprocessor()
        vectorizer, X,  = textPreprocessor.tfidf_vectorization(ls_df, factor_column_name) #get_X_vec(ls_df, factor_column_name)

        # Step 3: Split the labeled data into train and test sets (for evaluation)
        X_train, X_test, y_train, y_test = train_test_split(X[y != -1], y[y != -1], test_size=0.2, random_state=42)

        # Step 4: Create and fit the LabelSpreading model using ALL data (labeled + unlabeled)
        label_prop_model = LabelSpreading()
        label_prop_model.fit(X, y)

        # Step 5: Make predictions on the test set (only labeled data for evaluation)
        y_test_predict = label_prop_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_predict)
        print("Accuracy on test data:", accuracy)

        # Optionally show confusion matrix
        show_confusion_matrix(label_prop_model, X_test, y_test, label_encoder)

        # Step 6: Predict the labels for the full dataset (including unlabeled data)
        y_full_predict = label_prop_model.predict(X)

        # Handle -1 predictions by replacing them with 'Unmapped'
        y_full_predict = np.where(y_full_predict == -1, unmapped_encoder, y_full_predict)

        # Decode the predicted labels back to their original categories
        y_decoded = label_encoder.inverse_transform(y_full_predict)

        # Add the predicted labels back to the DataFrame
        ls_df['HFACS_Category_Value_Predict'] = y_decoded

        save_model_LS(label_prop_model, vectorizer, label_encoder, unmapped_encoder, ds_name)

    def predict(self, df: pd.DataFrame, sample_size=0):

        ds_name = self.ds_name
        factor_col_name = self.factor_col_name

        model, vectorizer, label_encoder, unmapped_encoder = load_model_LS(ds_name)

        # New set of data for prediction
        if sample_size > 0:
            labeled_data =  df.tail(sample_size).copy()
        else:
            labeled_data =  df

        # Drop rows with NaN in 'Combined_Factors'
        labeled_data = labeled_data.dropna(subset=[factor_col_name])

        # Vectorize the new set of data
        # X_vec = get_LS_X(labeled_data)
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
    #----------- Model Train/Predict Func---------