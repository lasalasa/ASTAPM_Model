# code adapted from (Abc — Abstract Base Classes, n.d.)
import os
import pandas as pd

from abc import ABC, abstractmethod

from src.extensions.pd_extension import PdExtension
from src.extensions.os_extension import OsOperation
from src.extensions.shutil_extension import FileOperation
from src.config import settings
from src.database import session_manager

class ETL(ABC):

    def __init__(self, data_source):
        ds_raw_db = data_source.ds_raw_db
        ds_name = data_source.ds_name

        self.ds_raw_db = ds_raw_db
        self.ds_name = ds_name

        self.ds_im_folder_path = data_source.ds_im_path
        self.ds_ex_folder_path = data_source.ds_ex_path

        ds_prefix = ds_name.lower()

        self.main_ex_folder_path = f"{settings.ex_folder_path}/{ds_prefix}"

        self.ds_prefix = ds_prefix
        self.narrative_table_name = f"{ds_prefix}_narrative"
        self.factors_table_name = f"{ds_prefix}_factors"

        self.main_pd_extension = PdExtension(settings.main_db_name)
        self.ds_pd_extension = PdExtension(ds_raw_db)

        self.transformed_data = {}
    
    @abstractmethod
    async def extract_data(self, source_name, im_folder_path, ex_folder_path):
        pass

    @abstractmethod
    def cleaned_data(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def build_main_mada(self, all_data):
        pass

    @abstractmethod
    def build_narrative_data(self, all_data):
        pass

    @abstractmethod
    def build_factors_data(self, all_data):
        pass

    async def transform_data(self):
        print("Transforming data...")
        ex_folder_path = self.main_ex_folder_path
        metadata = await session_manager.get_metadata(self.ds_raw_db)
        
        all_data = {}

        pd_Extension = self.ds_pd_extension

        os_operation = OsOperation()
        file_operation = FileOperation()

        file_operation.remove_folder(ex_folder_path)
        os_operation.create_folder(ex_folder_path)

        for table in metadata.tables.values():

            table_name:str = table.name

            df = pd_Extension.read_db(f"`{table_name}`")

            # Remove Rows and Cols where all values are NaN
            df_cleaned = self.cleaned_data(df)

            print(table_name, ' => df=', df.shape, ' vs ', 'df_cleaned=', df_cleaned.shape)

            if not df_cleaned.empty:
                table_name = table_name.lower().replace(' ', '_')
                new_table_name = f"{self.ds_prefix}_{table_name}"
                all_data[new_table_name] = df_cleaned

        self.build_data(all_data)

    async def load_data(self):
        print("Loading data...")
        # Save the main DataFrame
        main_df = self.transformed_data[self.ds_prefix]
        # self.save_data(main_df, self.ds_prefix)
        print("main_df=", main_df.shape)

        # Save the narrative DataFrame
        narrative_df = self.transformed_data[self.narrative_table_name]
        # Keep the first occurrence for each event_id in narrative_df and factors_df
        narrative_df = narrative_df.drop_duplicates(subset='event_id', keep='first')
        self.save_data(narrative_df, self.narrative_table_name)
        print("narrative_df=", narrative_df.shape)

        # Save the factors DataFrame
        factors_df = self.transformed_data[self.factors_table_name]
        factors_df = factors_df.drop_duplicates(subset='event_id', keep='first')
        self.save_data(factors_df, self.factors_table_name)
        print("factors_df=", factors_df.shape)

        combined_df = pd.merge(main_df, narrative_df, on='event_id', how='left')
        print("combined_df=", combined_df.shape)

        combined_df = pd.merge(combined_df, factors_df, on='event_id', how='left')
        print("combined_df=", combined_df.shape)

        combined_df = combined_df.dropna(subset=['narrative_01', 'narrative_02'], how='all')
        combined_df = combined_df.dropna(subset=['finding_description'])
        self.save_data(combined_df, self.ds_prefix)

    # Save data to both database and CSV
    def save_data(self, df, table_name):
        # Save to database
        self.main_pd_extension.save_to_db(df, table_name, "replace")

        # Save to CSV
        csv_store_file_path = os.path.join(self.main_ex_folder_path, f"{table_name}.csv")
        self.main_pd_extension.save_to_csv(df, csv_store_file_path)

    def build_data(self, all_data):
        self.transformed_data = {}
        self.transformed_data[self.ds_prefix] = self.build_main_mada(all_data)
        self.transformed_data[self.narrative_table_name] = self.build_narrative_data(all_data)
        self.transformed_data[self.factors_table_name] = self.build_factors_data(all_data)
        
# end of adapted code

