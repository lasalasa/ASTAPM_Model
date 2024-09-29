# Outer Service Module
import os
import pandas as pd

# Inner Service Module
from .etl import ETL
from src.extensions.pd_extension import PdExtension
from src.extensions.os_extension import OsOperation
from src.extensions.shutil_extension import FileOperation

# Standardize Time format
def get_time(time_range):
    if pd.notnull(time_range) and time_range != '':
        time_split = time_range.split('-')

        if len(time_split) == 2:
            from_time, to_time = time_split

            # Convert to HH:MM format
            from_time = f"{from_time[:2]}:{from_time[2:]}"
            to_time = f"{to_time[:2]}:{to_time[2:]}"
            # print(from_time, to_time)
            return from_time, to_time
        
        from_time = time_split
        from_time = f"{from_time[:2]}:{from_time[2:]}"
        # print(from_time)
        return from_time, None 
    else:
        return None, None

def combined_narrative(df):
    df['narrative_02'] = df['narrative_02'].fillna('')

    # Step 4: Combine 'narrative_01' and 'narrative_02' into a single 'narrative' column
    # code adapted from GeeksforGeeks (2021)
    df['narrative'] = df['narrative_01'].str.cat(df['narrative_02'], sep=' ', na_rep='')
    # end of adapted code
    return df

def counting_narrative(df):
    df['narrative_length'] = df['narrative'].apply(len)
    # Count words in each narrative
    df['narrative_word_count'] = df['narrative'].apply(lambda x: len(x.split()))

    # Count sentences in each narrative
    df['narrative_sentence_count'] = df['narrative'].apply(lambda x: len(x.split('. ')))
    
    return df

def clean_narrative(df):

    df = combined_narrative(df)
    df = counting_narrative(df)

    df.dropna(subset=['narrative'], inplace=True)

    df = df[df['narrative_word_count'] > 0].copy()
    return df

def clean_feature(df: pd.DataFrame, feature_name):
    df = df.dropna(subset=[feature_name])

    return df

def clean_event_factores(df):
    # df = df.fillna('')

    columns_to_check = ['contributing_factors', 'primary_problem','human_factors']
    df.dropna(thresh=len(columns_to_check) - 2, subset=columns_to_check, inplace=True)

    # Fill NaN in 'primary_problem' with the first item after splitting 'contributing_factors'
    df['primary_problem'] = df['primary_problem'].fillna(df['contributing_factors'].str.split(';').str[0])

    # Fill NaN in 'primary_problem' based on 'human_factors'
    df['primary_problem'] = df['primary_problem'].fillna('Human Factors')

    df['contributing_factors'] = df['contributing_factors'].fillna(df['primary_problem'])
    return df

# def combine_event_factors(row):
#     # # https://sparkbyexamples.com/pandas/pandas-concatenate-two-columns/
#     # # Concatenate the selected columns with a separator (e.g., a space or comma)
#     primary_problem = row['primary_problem']

#     # Check for 'Weather'
#     if primary_problem == 'Weather':
#         return primary_problem
    
#     # Check for 'Human Factors'
#     if primary_problem == 'Human Factors':
#         human_factors = row['human_factors'] if pd.notnull(row['human_factors']) else ''
#         anomaly = row['event_anomaly'] if pd.notnull(row['event_anomaly']) else ''
        
#         if human_factors and not human_factors == primary_problem:
#             return human_factors
#         elif human_factors and anomaly:
#             return f"{human_factors}:{anomaly}".rstrip(':')
#         return primary_problem

#     # For other primary problems
#     else:
#         human_factors = row['human_factors'] if pd.notnull(row['human_factors']) else ''
#         aircraft_component = row['aircraft_component'] if pd.notnull(row['aircraft_component']) else ''
#         problem = row['aircraft_component_problem'] if pd.notnull(row['aircraft_component_problem']) else ''
#         anomaly = row['event_anomaly'] if pd.notnull(row['event_anomaly']) else ''

#         contributing_factors = row['contributing_factors'] if pd.notnull(row['contributing_factors']) else ''
        
#         if human_factors and aircraft_component and problem and anomaly:
#             # print(f'----={primary_problem}>{human_factors}')
#             return f"{human_factors}:{aircraft_component}:{problem}:{anomaly}".rstrip(':')
        
#         if contributing_factors == 'Aircraft' and primary_problem == 'Aircraft':
#             return 'Aircraft-Aircraft'

#         contributing_factors = contributing_factors.replace("Aircraft", "")
#         if contributing_factors and aircraft_component and problem and anomaly:
#             return f"{primary_problem}:{aircraft_component}:{problem}:{anomaly}".rstrip(':')
#         return primary_problem

def define_finding_description(row):
    # # https://sparkbyexamples.com/pandas/pandas-concatenate-two-columns/
    # # Concatenate the selected columns with a separator (e.g., a space or comma)
    primary_problem = row['primary_problem']

    primary_problem:str = row['primary_problem'] if pd.notnull(row['primary_problem']) else ''
    contributing_factors:str = row['contributing_factors'] if pd.notnull(row['contributing_factors']) else ''
    human_factors:str = row['human_factors'] if pd.notnull(row['human_factors']) else ''
    problem = row['aircraft_component_problem'] if pd.notnull(row['aircraft_component_problem']) else ''
    anomaly = row['event_anomaly'] if pd.notnull(row['event_anomaly']) else ''

    return  "-".join(filter(lambda x: x != '', [primary_problem, contributing_factors, human_factors, problem, anomaly]))

    # return f"{primary_problem}-{contributing_factors}-{human_factors}-{problem}-{anomaly}".rstrip('-')

def get_human_factor(row, factor):
    human_factors:str = row['human_factors']
    
    if pd.isnull(human_factors):
        return factor
    
    human_factors_list = [factor.strip() for factor in human_factors.split(';')]
    return human_factors_list[0]

def define_finding_factor(row):
    factor = ''
    contributing_factors:str = row['contributing_factors']
    contributing_factors_list = [factor.strip() for factor in contributing_factors.split(';')]

    if len(contributing_factors_list) == 1:
        factor = contributing_factors_list[0]

        # if factor == 'Human Factors' or factor == 'Aircraft':
        factor = get_human_factor(row, factor)

        # if factor == 'Aircraft':
        #     return 'Aircraft-Aircraft systems'

    else:
        contributing_factors_list = [factor for factor in contributing_factors_list if factor != 'Aircraft']

        if len(contributing_factors_list) == 1:
            factor = contributing_factors_list[0]

            # if factor == 'Human Factors':
            factor = get_human_factor(row, factor)
        elif 'Human Factors' in contributing_factors_list:
            factor = contributing_factors_list[0]
            factor = get_human_factor(row, factor)
        else:
            factor = contributing_factors_list[0]

    return factor

class AsrsEtl(ETL):

    def __init__(self, data_source):
        super().__init__(data_source)
    
    async def extract_data(self):
        # https://www.geeksforgeeks.org/python-loop-through-folders-and-files-in-directory/

        # ds_raw_db = ds.ds_raw_db
        print("Extracting data...")

        im_folder_path = self.ds_im_folder_path
        ex_folder_path = self.ds_ex_folder_path

        os_operation = OsOperation()
        file_operation = FileOperation()

        pd_extension = self.ds_pd_extension

        file_name_list = os_operation.get_dir_files(im_folder_path)

        file_operation.remove_folder(ex_folder_path)
        os_operation.create_folder(ex_folder_path)

        for name in file_name_list:
            if name != ".DS_Store":
                file_path = os.path.join(im_folder_path, name)
                print(file_path)
                data = pd.read_csv(file_path)
            
                print(f"Reading... {file_path}")

                # grouped_columns = self.__group_columns(data)

                grouped_columns = {}

                for col in data.columns:
                    if col == ' ':
                        col = 'ACN'
                    prefix = col.split('.')[0]
                    if prefix not in grouped_columns:
                        grouped_columns[prefix] = []
                    grouped_columns[prefix].append(col)

                del grouped_columns['ACN']

                for prefix, columns in grouped_columns.items():
                    new_columns = [' '] + [col for col in columns]
                    grouped_data = data[new_columns]

                    # print(grouped_data)
                    new_header = grouped_data.iloc[0]
                    grouped_db_data = grouped_data[1:]
                    grouped_db_data.columns = new_header
                    grouped_db_data.reset_index(drop=True, inplace=True)
                    # print(grouped_db_data)
                    
                    csv_store_file_path = os.path.join(ex_folder_path, f"{prefix}.csv")

                    pd_extension.save_to_csv(grouped_db_data, csv_store_file_path)
        
        file_name_list = os_operation.get_dir_files(ex_folder_path)
        for name in file_name_list:
            file_path = os.path.join(ex_folder_path, name)
            
            data = pd.read_csv(file_path, low_memory=False)

            table_name = name.split('.')[0]

            print(f"Storing... {table_name}")
            pd_extension.save_to_db(data, table_name, "replace")

    def cleaned_data(self, df):

        # Return a new Index with elements of index not in other.
        columns_to_consider = df.columns.difference(['ACN'])

        # Remove rows where all values are NaN
        df_cleaned = df.dropna(how='all', subset=columns_to_consider)

        # Remove col where all values are NaN
        df_cleaned = df_cleaned.dropna(axis=1, how='all')

        return df_cleaned

    def build_main_mada(self, raw_data):

        table_name = self.ds_prefix
        data_df = pd.DataFrame(raw_data[f'{table_name}_time'])[['ACN', 'Date', 'Local Time Of Day']].copy()

        # Merge with 'asrs_place'
        place_data = f'{table_name}_place'
        if place_data in raw_data:
            structural_df = pd.merge(data_df,
                raw_data[place_data][['ACN', 'Locale Reference', 'State Reference']],
                on='ACN', how='left')
            structural_df.rename(columns={'Locale Reference': 'place_local_ref', 'State Reference': 'place_state_ref'}, inplace=True)
        
        data_df.rename(columns={'ACN': 'event_id', 'Date': 'date', 'Local Time Of Day': 'time'}, inplace=True)

        data_df['date'] = pd.to_datetime(data_df['date'], format='%Y%m', errors='coerce')
        data_df['year'] = data_df['date'].dt.year
        data_df['month'] = data_df['date'].dt.month

        # Bind from_time and to_time  to data
        data_df[['from_time', 'to_time']] = data_df['time'].apply(lambda x: pd.Series(get_time(x)))
        
        # Remove the 'time' column
        data_df.drop('time', axis=1, inplace=True)

        return data_df

    def build_narrative_data(self, raw_data):
        table_name = self.ds_prefix

        data_df = pd.DataFrame(raw_data[f'{table_name}_time'])[['ACN']].copy()

        data_df = pd.merge(data_df,
            raw_data[f'{table_name}_report_1'][['ACN', 'Narrative']],
            on='ACN', how='left')

        data_df = pd.merge(data_df,
            raw_data[f'{table_name}_report_2'][['ACN', 'Narrative']],
            on='ACN', how='left',
            suffixes=('_01', '_02'))
        
        data_df.rename(columns={'ACN': 'event_id', 'Narrative_01': 'narrative_01', 'Narrative_02': 'narrative_02'}, inplace=True)

        data_df = clean_narrative(data_df)

        data_df = data_df.drop('narrative', axis=1)

        return data_df

    def build_factors_data(self, all_data):
        table_name = self.ds_prefix

        data_df = pd.DataFrame(all_data[f'{table_name}_time'])[['ACN']].copy()

        # Merge with 'asrs_component'
        component = f'{table_name}_component'
        if component in all_data:
            data_df = pd.merge(data_df,
                all_data[component][['ACN', 'Aircraft Component', 'Problem']],
                on='ACN', how='left')
            data_df.rename(columns={'Aircraft Component': 'aircraft_component', 'Problem': 'aircraft_component_problem'}, inplace=True)

        # Merge with 'asrs_environment'
        environment = f'{table_name}_environment'
        if environment in all_data:
            data_df = pd.merge(data_df,
                all_data[environment][['ACN', 'Flight Conditions', 'Weather Elements / Visibility', 'Light', 'Work Environment Factor']],
                on='ACN', how='left')
            
            data_df.rename(columns={
                'Flight Conditions': 'environment_flight_conditions',
                'Weather Elements / Visibility': 'environment_weather_elements_visibility',
                'Light': 'environment_light', 'Work Environment Factor': 'work_environment_factor'}, inplace=True)

        # Merge with 'asrs_assessments'
        assessments = f'{table_name}_assessments'
        if 'asrs_assessments' in all_data:
            data_df = pd.merge(data_df,
                all_data[assessments][['ACN', 'Contributing Factors / Situations', 'Primary Problem']],
                on='ACN', how='left')
            data_df.rename(columns={
                'Contributing Factors / Situations': 'contributing_factors',
                'Primary Problem': 'primary_problem'}, inplace=True)

        # Merge with 'asrs_person_1'
        person_1 = f'{table_name}_person_1'
        if person_1 in all_data:
            data_df = pd.merge(data_df,
                all_data[person_1][['ACN', 'Human Factors', 'Communication Breakdown']],
                on='ACN', how='left')
            data_df.rename(columns={'Human Factors': 'human_factors', 'Communication Breakdown': 'communication_breakdown'}, inplace=True)

        # Merge with 'asrs_event'
        event = f'{table_name}_events'
        if person_1 in all_data:
            data_df = pd.merge(data_df, all_data[event][['ACN', 'Anomaly']], on='ACN', how='left')
            data_df.rename(columns={'Anomaly': 'event_anomaly'}, inplace=True)

        data_df.rename(columns={'ACN': 'event_id'}, inplace=True)

        data_df = clean_event_factores(data_df)

        # data_df['finding_description'] = data_df.apply(combine_event_factors, axis=1)

        data_df['finding_factor'] = data_df.apply(define_finding_factor ,  axis=1)
        data_df['finding_description'] = data_df.apply(define_finding_description ,  axis=1)

        data_df = clean_feature(data_df, 'finding_description')

        return data_df


    
