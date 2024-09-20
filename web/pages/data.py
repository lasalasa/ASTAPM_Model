import os
import pandas as pd

FOLDER_PATH_DATA_STORE = "web/data_store"
DATA_STORE_TYPES = ["csv"]
DATA_SOURCES = ["asrs", "ntsb"]

data_type = DATA_STORE_TYPES[0]
source_name = DATA_SOURCES[0]
unique_id = "20240613175806"

folder_path = os.path.join(FOLDER_PATH_DATA_STORE, f"{source_name}_store",data_type, unique_id)
file_name_list_unsorted = os.listdir(folder_path)
file_name_list = sorted(file_name_list_unsorted)

# name_list = [file_name.replace(' ', '_').lower().split('.')[0] for file_name in file_name_list]



# asrs_data = {}

# for i, row in enumerate(file_name_list):
#     key = name_list[i]
#     asrs_data[key] = pd.read_csv(os.path.join(folder_path, row), low_memory=False)

# asrs_whole_data = pd.DataFrame()
# asrs_whole_data['ACN'] = asrs_data['aircraft_1']['ACN']

# for key in asrs_data.keys():
#     df = asrs_data[key].copy()
#     prefix = f'{key}__'
#     df.columns = [prefix + col if not col == 'ACN' else col for col in df.columns]
    
#     new_df =  df[[col for col in df.columns if col != 'ACN']]
#     asrs_whole_data = pd.concat([asrs_whole_data, new_df], axis=1)

# df = asrs_whole_data.copy()
# null_df = df.isnull().mean() * 100
# columns_to_drop = null_df[null_df == 100].index.tolist()

# asrs_whole_cleaned_data = df.drop(columns=columns_to_drop)

# df = asrs_whole_cleaned_data.copy()

# Count plot for categorical columns
# df_unique = asrs_whole_cleaned_data.nunique()
# categorical_columns = df.select_dtypes(include=['object']).columns
# for column in categorical_columns:
#     df[column] = df[column].str.split(';')
#     contributing_factors = df[column].explode().str.strip()

#     # Count the occurrences of each contributing factor
#     contributing_factors_summary = contributing_factors.value_counts()

#     unique_count = contributing_factors.nunique()

#     if unique_count < 100:
#         print(column, unique_count)
#         col_list = column.split('__')

#         print(col_list)
#         head_name = col_list[0]
#         attribute_name = col_list[1]

#         # Plot the top 10 contributing factors
#         top_contributing_factors = contributing_factors_summary.head(10)

#         plt.figure(figsize=(12, 6))
#         top_contributing_factors.plot(kind='bar')
#         plt.title(f'Top 10 {head_name} {attribute_name} Factors to Aviation Incidents')
#         plt.xlabel(f'{head_name} {attribute_name} Factor')
#         plt.ylabel('Number of Incidents')
#         plt.xticks(rotation=45)
#         plt.show()