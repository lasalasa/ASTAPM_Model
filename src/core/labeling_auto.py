import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.core.core_mappers import HFACS_DICTIONARY, HFACS_MAPPING_DICTIONARY
from src.core.core_utils import CoreUtils

def get_dic(is_balance):
    if is_balance:
       return HFACS_MAPPING_DICTIONARY['hfacs_mapping_balance']
    else:
       return HFACS_MAPPING_DICTIONARY['hfacs_mapping']
    
def show_disctribution(df):

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html#sphx-glr-gallery-subplots-axes-and-figures-align-labels-demo-py
    # Create a figure with 1 row and 2 columns of subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the first countplot on the first axes
    sns.countplot(x='HFACS_Category_Value', data=df, ax=axes[0])
    axes[0].set_title('The distribution of Taxonomy (With Original HFACS)')
    axes[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels if needed

    # Plot the second countplot on the second axes
    sns.countplot(x='HFACS_Category_balance_Value', data=df, ax=axes[1])
    axes[1].set_title('The distribution of Taxonomy (With Proposed HFACS)')
    axes[1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels if needed

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

class AutoLabeling:

    def __init__(self, df):

        factor_col_name = CoreUtils.get_constant()["FACTOR_COL_NAME"]
        
        self.factor_col_name = factor_col_name
       
        self.df = df
    
    def classify_document(self, doc_text, is_imbalance):
       
        hfacs_categories = get_dic(is_imbalance)

        for category, subcategories in hfacs_categories.items():
            for subcategory, keywords in subcategories.items():
                
                for keyword in keywords:

                    # First check for an exact match
                    if keyword.lower() == doc_text.lower():
                        return HFACS_DICTIONARY[subcategory]
                    
                    # If no exact match, check if the keyword is present as a substring
                    elif keyword.lower() in doc_text.lower():
                        return HFACS_DICTIONARY[subcategory]
                    
        #'Unknown'  # If no category matches
        return HFACS_DICTIONARY[-1]

    def do_auto_label(self, sample_size=0):

        factor_col_name = self.factor_col_name
        df = self.df
        df_size = df.shape[0]

        if sample_size > 0 and df_size > sample_size:
            df_tail =  df.sample(n=sample_size, random_state=42)
        else:
            df_tail = df
            
        print('AutoLabeling sample_size=', df_tail.shape)
        
        df_tail.replace('', pd.NA, inplace=True)

        data = df_tail.copy()

        data['HFACS_Category'] = data[factor_col_name].apply(lambda x: self.classify_document(x, False))
        data['HFACS_Category_Level'] = data['HFACS_Category'].apply(lambda x: x[0])
        data['HFACS_Category_Value'] = data['HFACS_Category'].apply(lambda x: f"{x[0]}-{x[3]}".rstrip('-'))

        data['HFACS_Category_balance'] = data[factor_col_name].apply(lambda x: self.classify_document(x, True))
        data['HFACS_Category_balance_Level'] = data['HFACS_Category_balance'].apply(lambda x: x[0])
        data['HFACS_Category_balance_Value'] = data['HFACS_Category_balance'].apply(lambda x: f"{x[0]}-{x[3]}".rstrip('-'))

        show_disctribution(data)

        return data