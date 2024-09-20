import os
import pandas as pd

from .etl import ETL

from src.extensions.pd_extension import PdExtension
from src.extensions.os_extension import OsOperation
from src.extensions.shutil_extension import FileOperation

from src.config import settings
from src.database import session_manager

# TODO should move to ETL
# df.fillna('', inplace=True)
# df[df['contributing_factors'].str.contains(df['primary_problem'])==False]
# columns_to_check = ['contributing_factors', 'primary_problem','human_factors']
# df.dropna(thresh=len(columns_to_check) - 2, subset=columns_to_check, inplace=True)

# Fill NaN in 'primary_problem' with the first item after splitting 'contributing_factors'
# df['primary_problem'] = df['primary_problem'].fillna(df['contributing_factors'].str.split(';').str[0])

# Fill NaN in 'primary_problem' based on 'human_factors'
# df['primary_problem'] = df['primary_problem'].fillna('Human Factors')

# df['contributing_factors'] = df['contributing_factors'].fillna(df['primary_problem'])
# df['human_factors'] = df['human_factors'].fillna('no_hf')

# Function to handle splitting and conversion
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

def combine_event_factors(row):
    # # https://sparkbyexamples.com/pandas/pandas-concatenate-two-columns/
    # # Concatenate the selected columns with a separator (e.g., a space or comma)
    primary_problem = row['primary_problem']

    # Check for 'Weather'
    if primary_problem == 'Weather':
        return primary_problem
    
    # Check for 'Human Factors'
    if primary_problem == 'Human Factors':
        human_factors = row['human_factors'] if pd.notnull(row['human_factors']) else ''
        anomaly = row['event_anomaly'] if pd.notnull(row['event_anomaly']) else ''
        
        if human_factors and anomaly:
            return f"{primary_problem}:{human_factors}:{anomaly}".rstrip(':')
        return primary_problem

    # For other primary problems
    else:
        aircraft_component = row['aircraft_component'] if pd.notnull(row['aircraft_component']) else ''
        problem = row['aircraft_component_problem'] if pd.notnull(row['aircraft_component_problem']) else ''
        anomaly = row['event_anomaly'] if pd.notnull(row['event_anomaly']) else ''
        
        if aircraft_component and problem and anomaly:
            return f"{primary_problem}:{aircraft_component}:{problem}:{anomaly}".rstrip(':')
        return primary_problem

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

    # async def transform_data(self):
    #     print("Transforming data...")
    #     ex_folder_path = self.main_ex_folder_path
    #     metadata = await session_manager.get_metadata(self.ds_raw_db)
        
    #     all_data = {}

    #     main_pd_Extension = self.main_pd_extension
    #     pd_Extension = self.ds_pd_extension

    #     os_operation = OsOperation()
    #     file_operation = FileOperation()

    #     file_operation.remove_folder(ex_folder_path)
    #     os_operation.create_folder(ex_folder_path)

    #     for table in metadata.tables.values():

    #         table_name:str = table.name

    #         df = pd_Extension.read_db(f"`{table_name}`")

    #         # Remove rows/cols where all values are NaN
    #         df_cleaned = self.cleaned_data(df)

    #         print('df=', df.size, ' vs ', 'df_cleaned=', df_cleaned.size)

    #         if not df_cleaned.empty:
    #             table_name = table_name.lower().replace(' ', '_')
    #             new_table_name = f"{self.ds_prefix}_{table_name}"
    #             all_data[new_table_name] = df_cleaned
    #             # main_pd_Extension.save_to_db(df_cleaned, new_table_name)

    #     self.build_data(all_data)
    
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

        data_df['finding_description'] = data_df.apply(combine_event_factors, axis=1)
        return data_df


    
