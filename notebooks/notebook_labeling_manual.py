# %%
%run notebook_core_utils.ipynb
%run notebook_core_mappers.ipynb
%run notebook_labeling_auto.ipynb


# %%

ntsb_df = CoreUtils.get_data('ntsb', from_year=2015, to_year=2023)
asrs_ntsb_df =  pd.concat([asrs_df, ntsb_df], axis=0).reset_index(drop=True)

# %%
import pandas as pd
PATH_PREFIX = '../data/local_ex/astapm'
# asrs_df = CoreUtils.get_data('asrs', from_year=2022, to_year=2023)
asrs_df = pd.read_csv(f'{PATH_PREFIX}/asrs/asrs.csv', low_memory=False)
asrs_df = asrs_df[(asrs_df['year'] >= 2022) & (asrs_df['year'] <= 2023)]

# %%
df = asrs_df[['event_id', 'primary_problem', 'contributing_factors', 'human_factors']].copy()
df['primary_problem'].value_counts()

# df[df['primary_problem'] == 'Human Factors']

# %%
df[df['primary_problem'] == 'Human Factors'].sample(n=10, random_state=42)

# %%
df['human_factors'].str.split(';', expand=True)

# %%
df[df['primary_problem'] == 'Aircraft'].sample(n=10, random_state=42)

# %%
def define_finding_description(row):
    # # https://sparkbyexamples.com/pandas/pandas-concatenate-two-columns/
    # # Concatenate the selected columns with a separator (e.g., a space or comma)
    primary_problem = row['primary_problem']

    primary_problem:str = row['primary_problem']
    contributing_factors:str = row['contributing_factors']
    human_factors:str = row['human_factors']

    return f"{primary_problem}-{contributing_factors}-{human_factors}".rstrip(':')

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

        if factor == 'Aircraft':
            return 'Aircraft-Aircraft systems'

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

df['finding_factor'] = df.apply(define_finding_factor ,  axis=1)
df['finding_description'] = df.apply(define_finding_description ,  axis=1)

df['finding_factor'].value_counts()

# %%
df[['finding_factor', 'finding_description']]

# %%
# asrs_df = CoreUtils.get_data('asrs', from_year=2022, to_year=2023)
# sample_df =  asrs_df.copy() #.sample(n=2000)

autoLabeling = AutoLabeling(df)

data = autoLabeling.do_auto_label()

print(data.shape)
data[['finding_description', 'HFACS_Category_balance_Value']].to_csv('./test.csv')

# %% [markdown]
# # NTSB

# %%
import pandas as pd
PATH_PREFIX = '../data/local_ex/astapm'
# asrs_df = CoreUtils.get_data('asrs', from_year=2022, to_year=2023)
ntsb_df = pd.read_csv(f'{PATH_PREFIX}/ntsb/ntsb.csv', low_memory=False)
ntsb_df = ntsb_df[(ntsb_df['year'] >= 2018) & (ntsb_df['year'] <= 2023)]


# %%

n_df = ntsb_df.copy()

def define_finding_factor_n(row):
    factor = ''
    finding_description:str = row['finding_description']
    finding_description_list = [factor.strip() for factor in finding_description.split('-')]

    if finding_description_list[0] == 'Not determined':
        return 'Not determined'
    
    factor = f"{finding_description_list[0]}-{finding_description_list[1]}-{finding_description_list[2]}".rstrip('-')

    return factor

n_df['finding_factor'] = n_df.apply(define_finding_factor_n ,  axis=1)
n_df = n_df[n_df['finding_factor'] != 'Not determined']

n_df['finding_factor'].value_counts()

# %%
n_df[['finding_factor', 'finding_description']]

# %%
# sample_df =  ntsb_df.copy() #.sample(n=2000)

autoLabeling = AutoLabeling(n_df)

data = autoLabeling.do_auto_label()
# print(sample_df.shape)


# %%
data[data['HFACS_Category_Value'] == 'Unmapped']

# %%

sample_df =  asrs_ntsb_df.copy() #.sample(n=2000)

autoLabeling = AutoLabeling(sample_df)

autoLabeling.do_auto_label()

print(sample_df.shape)



# %%
sample_df[sample_df['finding_description']=='Aircraft-Aircraft']


