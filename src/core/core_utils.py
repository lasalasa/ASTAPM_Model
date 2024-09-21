import pandas as pd

# Variables
FACTOR_COL_NAME = "finding_description"

factors_columns = [ FACTOR_COL_NAME ]

narrative_columns = ['narrative_01', 'narrative_02']

filtered_columns = ['event_id'] + factors_columns + narrative_columns

PATH_PREFIX = '../../data/local_ex/astapm'

class CoreUtils:
    #----------- Data Manipulation Func ----------

    @staticmethod
    def get_constant():
        return {
            "FACTOR_COL_NAME": FACTOR_COL_NAME,
        }
    
    @staticmethod
    def get_factor_cols():
        return factors_columns
    
    @staticmethod
    def get_narrative_cols():
        return narrative_columns

    @staticmethod
    def get_filtered_cols():
        return filtered_columns

    @staticmethod
    def get_data(ds_name, from_year=2020, to_year=2023):

        path_prefix = PATH_PREFIX

        data_df = pd.read_csv(f'{path_prefix}/{ds_name}/{ds_name}.csv', low_memory=False)
        # asrs_narrative = pd.read_csv(f'{path_prefix}/{ds_name}/{ds_name}_narrative.csv', low_memory=False)
        # asrs_factors = pd.read_csv(f'{path_prefix}/{ds_name}/{ds_name}_safety_factors.csv', low_memory=False)

        # data_df = pd.merge(asrs, asrs_narrative, on='event_id', how='left')
        # data_df = pd.merge(data_df, asrs_factors, on='event_id', how='left')

        data_df = data_df[(data_df['year'] >= from_year) & (data_df['year'] <= to_year)]

        df = data_df[filtered_columns].copy()
        return df