import os
import shutil
import csv
import sys

import pandas as pd
from datetime import datetime

class OsOperation:
    def __init__(self):
        pass
    
    def create_folder(self, folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(f"An error occurred (create_folder): {e}")

    # https://www.geeksforgeeks.org/python-loop-through-folders-and-files-in-directory/
    def get_dir_files(self, folder_path):
        try:
            file_name_list = os.listdir(folder_path)
            sorted_file_name_list = sorted(file_name_list)
            return sorted_file_name_list
        except Exception as e:
            print(f"An error occurred (listdir_files): {e}")

    def get_dir_unique_path(self, folder_path):

        unique_name = datetime.now().strftime('%Y%m%d%H%M%S')

        unique_path = os.path.join(folder_path, unique_name)

        return unique_path