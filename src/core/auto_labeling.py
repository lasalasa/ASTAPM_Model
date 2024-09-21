import pandas as pd
from .core_mappers import HFACS_DICTIONARY, AUTO_LABELING_DICTIONARY, AST_APM_DICTIONARY
from .core_utils import CoreUtils

def get_dic(ds_name, imbalance_options):
    is_merge_taxonomy = imbalance_options["is_merge_taxonomy"]
    if is_merge_taxonomy:
        return [
            # HFACS_DICTIONARY['hfacs_balance'],
            AST_APM_DICTIONARY,
            AUTO_LABELING_DICTIONARY[f'{ds_name}_balance']
        ]
    else:
        return [
            HFACS_DICTIONARY['hfacs'],
            AUTO_LABELING_DICTIONARY[ds_name]
        ]

class AutoLabeling:

    def __init__(self, df, ds_name='asrs', imbalance_options={
        "is_merge_taxonomy": False
    }):
        self.ds_name = ds_name
        self.imbalance_options = imbalance_options

        factor_col_name = CoreUtils.get_constant()["FACTOR_COL_NAME"]
        hfacs_dic, auto_label_dic = get_dic(ds_name, imbalance_options)
        
        self.factor_col_name = factor_col_name
        self.hfacs_dic = hfacs_dic
        self.auto_label_dic = auto_label_dic
        self.df = df

    # Function to map combined issues to HFACS based on keyword mappings
    def __map_hfacs(self, factors):

        pf_dic = self.auto_label_dic

        # Convert to lowercase for consistent keyword matching
        factors = str(factors).lower()

        for keyword, mapping in pf_dic.items():
            if keyword.lower() in factors:  # Check if keyword is in the combined issues string
                # print(keyword.lower(), '=>>>', factors)
                # if(mapping == 11):
                #     # print(mapping)
                #     return self.__map_hf_asrs(factors)
                return mapping
        return -1  # Default to unmapped if no match found

    def do_auto_label(self, sample_size=0):

        # df_tail = df.tail(sample_size).copy()

        factor_col_name = self.factor_col_name
        df = self.df
        hfacs_dic = self.hfacs_dic

        if sample_size > 0:
            df_tail =  df.sample(n=sample_size, random_state=42)
        else:
            df_tail = df
            
        print('sample_size=', df_tail.shape)
        
        df_tail.replace('', pd.NA, inplace=True)
        # print_column = ['ACN'] + factors_column

        # df_tail[print_column].value_counts()
        # df_tail.isnull().sum()

        # Load the dataset (update the file path as needed)
        data = df_tail.copy()
        # Apply the function to each 'Combined_Issues' row and map to HFACS
        data['HFACS_Category_Index'] = data[factor_col_name].apply(self.__map_hfacs)

        # Split the 'HFACS_Mapped' into two new columns: 'HFACS_Category' and 'HFACS_Level'
        data['HFACS_Category_Value'] = data['HFACS_Category_Index'].apply(lambda x: hfacs_dic[x][3])
        data['HFACS_Level'] = data['HFACS_Category_Index'].apply(lambda x: hfacs_dic[x][0])

        # # Summarize the mapped HFACS categories and levels
        hfacs_summary = data.groupby(['HFACS_Category_Value']).size().reset_index(name='Count')
        print(hfacs_summary)
        return data