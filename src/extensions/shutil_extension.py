import shutil

class FileOperation:
    def __init__(self):
        pass

    def remove_folder(self, folder_path):
        try:
             # Remove the directory and all its contents
            shutil.rmtree(folder_path)
            print(f"Directory '{folder_path}' has been removed successfully.")
        except Exception as e:
            print(f"An error occurred (create_folder): {e}")