import os
import pandas as pd

from src.extensions.os_extension import OsOperation
from src.extensions.pd_extension import PdExtension
from src.extensions.shutil_extension import FileOperation

from src.extensions.mdb_parser_extension import MdbParserExtension

from .etl import ETL

from src.config import settings
from src.database import session_manager

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
    #             # main_pd_Extension.save_to_db(df_cleaned, f"{ds_name.lower()}_{new_table_name}", 'replace')

    #     self.build_data(all_data)

    # async def load_data(self):
    #     print("Loading data...")
    #     # Save the main DataFrame
    #     main_df = self.transformed_data[self.ds_prefix]
    #     self.save_data(main_df, self.ds_prefix)

    #     # Save the narrative DataFrame
    #     narrative_df = self.transformed_data[self.narrative_table_name]
    #     self.save_data(narrative_df, self.narrative_table_name)

    #     # Save the factors DataFrame
    #     factors_df = self.transformed_data[self.factors_table_name]
    #     self.save_data(factors_df, self.factors_table_name)

    def build_main_mada(self, raw_data):

        table_name = self.ds_prefix
        data_df = pd.DataFrame(raw_data[f'{table_name}_events'])[['ev_id', 'ev_date', 'ev_year', 'ev_month', 'ev_country', 'ev_state']].copy()

        data_df.rename(columns={'ev_id': 'event_id', 'ev_date': 'date', 'ev_year': 'year', 'ev_month': 'month', 'ev_country': 'country', 'ev_state': 'state'}, inplace=True)

        # https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        data_df['date'] = pd.to_datetime(data_df['date'], format='%d/%m/%y %H:%M:%S', errors='coerce')

        print(data_df['date'].head())

        return data_df

    def build_narrative_data(self, raw_data):

        table_name = self.ds_prefix

        data_df = pd.DataFrame(raw_data[f'{table_name}_events'])[['ev_id']].copy()

        data_df = pd.merge(data_df,
                        raw_data[f'{table_name}_narratives'][['ev_id', 'narr_accp', 'narr_accp_2']],
                        on='ev_id', how='left')
        
        data_df.rename(columns={'ev_id': 'event_id', 'narr_accp': 'narrative_01', 'narr_accp_2': 'narrative_02'}, inplace=True)

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

        return data_df

