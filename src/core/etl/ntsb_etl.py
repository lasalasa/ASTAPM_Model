# Outer Service Module
import os
import pandas as pd

#Inner Service module
from src.extensions.os_extension import OsOperation
from src.extensions.shutil_extension import FileOperation
from src.extensions.mdb_parser_extension import MdbParserExtension
from .etl import ETL


def combined_narrative(df):
    df['narrative_02'] = df['narrative_02'].fillna('')

    # Step 4: Combine 'narrative_01' and 'narrative_02' into a single 'narrative' column
    # code adapted from GeeksforGeeks (2021)
    df['narrative'] = df['narrative_01'].str.cat(df['narrative_02'], sep=' ', na_rep=None)

    df = df.dropna(subset=['narrative'])

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

    df = df[df['narrative_word_count'] > 0].copy()
    return df

def clean_feature(df: pd.DataFrame, feature_name):
    df = df.dropna(subset=[feature_name])

    return df

def define_finding_factor(row):
    factor = ''
    finding_description:str = row['finding_description']
    finding_description_list = [factor.strip() for factor in finding_description.split('-')]

    if finding_description_list[0] == 'Not determined':
        return 'Not determined'
    
    factor = f"{finding_description_list[0]}-{finding_description_list[1]}-{finding_description_list[2]}".rstrip('-')

    return factor

class NtsbEtl(ETL):

    def __init__(self, data_source):
        super().__init__(data_source)

    async def extract_data(self):
        
        print("Extracting data...")

        im_folder_path = self.ds_im_folder_path
        ex_folder_path = self.ds_ex_folder_path

        os_operation = OsOperation()
        file_operation = FileOperation()

        pd_extension = self.ds_pd_extension

        # file_name_list = os_operation.get_dir_files(im_folder_path)

        file_operation.remove_folder(ex_folder_path)
        os_operation.create_folder(ex_folder_path)

        # db_parser = NtsbParser()
        db_parser = MdbParserExtension(f"{im_folder_path}/avall.mdb")

        tables = db_parser.get_tables()

        error_tables = []
        
        file_operation.remove_folder(ex_folder_path)
        os_operation.create_folder(ex_folder_path)

        # Iterate over each table and save data to Mysql
        for table_name in tables:
            try:
                # Read data from Access table into a pandas DataFrame
                table = db_parser.read_table(table_name)

                # Extract the column names
                column_names = [column for column in table.columns]

                # Extract the data as a list of dictionaries
                data = []
                for row in table:
                    row_data = {column_names[i]: row[i] for i in range(len(row))}

                    if table_name == "narratives":
                        if(len(row_data['narr_accp']) > 65535):
                            narr_accp = row_data['narr_accp']
                            if isinstance(narr_accp, str) and len(narr_accp) > 65535:
                                new_length = round(len(narr_accp)/2)
                                row_data['narr_accp'] = narr_accp[:new_length]
                                row_data['narr_accp_2'] = narr_accp[new_length:]
                            else:
                                row_data['narr_accp_2'] = None
                            print(row_data['ev_id'], len(row_data['narr_accp']), len(row_data['narr_accp_2']))
                    data.append(row_data)

                df = pd.DataFrame(data)

                csv_store_file_path = os.path.join(ex_folder_path, f"{table_name}.csv")

                print("Storing...", csv_store_file_path)
                if not df.empty:
                    pd_extension.save_to_csv(df, csv_store_file_path)
                    pd_extension.save_to_db(df, table_name, "replace")

                print(f"{table_name} migration completed successfully")

            except Exception as e:
                error_tables.append(table_name)
                print(f'\n\n ############## ERROR {table_name} ##############' )
                print(e)
                print(f'\n\n ############## ############## ############## \n\n' )

    def cleaned_data(self, df):

        # Return a new index with elements of the index that are not in the other.
        columns_to_consider = df.columns.difference(['event_id'])

        # Remove rows where all values are NaN
        df_cleaned = df.dropna(how='all', subset=columns_to_consider)

        # Remove col where all values are NaN
        df_cleaned = df_cleaned.dropna(axis=1, how='all')

        return df_cleaned
    
    def build_main_mada(self, raw_data):

        table_name = self.ds_prefix
        data_df = pd.DataFrame(raw_data[f'{table_name}_events'])[['ev_id', 'ev_date', 'ev_year', 'ev_month', 'ev_country', 'ev_state']].copy()

        data_df.rename(columns={'ev_id': 'event_id', 'ev_date': 'date', 'ev_year': 'year', 'ev_month': 'month', 'ev_country': 'country', 'ev_state': 'state'}, inplace=True)

        # Assuming year and month are integer columns
        data_df['year'] = data_df['year'].astype(int)
        data_df['month'] = data_df['month'].astype(int)

        # Create a date column, assuming day = 1 (as day is not provided)
        data_df['date'] = pd.to_datetime(data_df[['year', 'month']].assign(day=1))

        # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        # data_df['date'] = pd.to_datetime(data_df['date'], format='%d/%m/%y %H:%M:%S', errors='coerce')

        print(data_df['date'].head())

        return data_df

    def build_narrative_data(self, raw_data):

        table_name = self.ds_prefix

        data_df = pd.DataFrame(raw_data[f'{table_name}_events'])[['ev_id']].copy()

        raw_df = raw_data[f'{table_name}_narratives'][['ev_id', 'narr_accp', 'narr_accp_2']].copy()

        print(raw_df.isnull().sum())

        data_df = pd.merge(data_df, raw_df, on='ev_id', how='left')

        print("==================")

        print(data_df.isnull().sum())
        
        data_df.rename(columns={'ev_id': 'event_id', 'narr_accp': 'narrative_01', 'narr_accp_2': 'narrative_02'}, inplace=True)

        data_df = clean_narrative(data_df)

        data_df = data_df.drop('narrative', axis=1)

        return data_df

    def build_factors_data(self, all_data):
        
        table_name = self.ds_prefix

        data_df = pd.DataFrame(all_data[f'{table_name}_events'])[['ev_id']].copy()

        # Merge with 'asrs_component'
        component = f'{table_name}_findings'
        if component in all_data:
            data_df = pd.merge(data_df,
                all_data[component][['ev_id', 'finding_description']],
                on='ev_id', how='left')
            # TODO
            # data_df.rename(columns={'finding_description': 'primary_problem'}, inplace=True)

        data_df.rename(columns={'ev_id': 'event_id'}, inplace=True)

        data_df = clean_feature(data_df, 'finding_description')

        data_df['finding_factor'] = data_df.apply(define_finding_factor,  axis=1)
        data_df = data_df[data_df['finding_factor'] != 'Not determined']

        return data_df

