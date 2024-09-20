import os
import csv
import sys

import pandas as pd
from sqlalchemy import text, create_engine

# https://www.geeksforgeeks.org/python-import-from-parent-directory/
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings

class PdExtension:

    def __init__(self, db_name):
        self.pd = pd
        self.db_name = db_name

    @staticmethod
    def get_engine(db_name):
        try:
            base_db_conn = settings.base_db_conn
            conn = f"{base_db_conn}/{db_name}"
            engine = create_engine(conn)
            
            return engine
        except Exception as e:
            print(e)
            print(f"Error connecting to database: {e}")
            print("Error occurred when creating engine: ", sys.exc_info()[0])
            exit()

    def save_to_csv(self, df: pd.DataFrame, file_path):
        if not os.path.isfile(file_path):
            df.to_csv(file_path, mode='w', header=True, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print (f"CSV file written... {file_path}")
        else:
            df.to_csv(file_path, mode='a', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
            print (f"CSV file appended... {file_path}")

    # Export to JSON
    def save_to_json(self, df, filename):
        df_json = df.to_json(orient='records')

        # filename_json = 'artists.json'
        with open(filename, 'w') as f:
            f.write(df_json)
        print ("JSON file written")

    # Insert new values to the existing table.
    def save_to_db(self, df: pd.DataFrame, table_name, if_exists='append'):
        try: 
            engine = PdExtension.get_engine(self.db_name)

            df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
            # if engine.has_table(table_name):
            #     # Table exists, append data
            #     df.to_sql(table_name, con=engine, if_exists='append', index=False)
            # else:
            #     # Table does not exist, create table
            #     df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        except Exception as e:
            print(e)
            print("Error occurred when saving to db: ", sys.exc_info()[0])
            exit()

    def read_db(self, table_name: str):
        try: 
            # Assuming PdReadExtension.get_engine(self.db_name) returns a valid SQLAlchemy engine
            engine = PdExtension.get_engine(self.db_name)

            # Read SQL table into a DataFrame
            df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
            
            return df
        except Exception as e:
            print(f"Error occurred when reading from db: {e}")
            print("Exception type:", sys.exc_info()[0])
            raise e
        
    def drop_all(self, df: pd.DataFrame, column_to_ignore: str):

        columns_to_consider = df.columns.difference([column_to_ignore])

        # If you want to remove rows where all values are NaN
        df_cleaned = df.dropna(how='all', subset=columns_to_consider)
        df_cleaned = df_cleaned.dropna(axis=1, how='all')

        return df_cleaned