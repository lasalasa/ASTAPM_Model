# %% [markdown]
# # Taxonomy Classification Model (Label Spreading)

# %% [markdown]
# ## Import Notebook

# %%
%run notebook_model_ls.ipynb
%run notebook_model_lstm.ipynb

# %% [markdown]
# ## ASRS Data Source

# %%
options={ "is_merge_taxonomy": True }

# 01. Get Data
asrs_df = CoreUtils.get_data('asrs', from_year=2000, to_year=2023)

# 02. Label Spreading
asrs_modelLS = ModelLS({'asrs': asrs_df}, 'asrs', 10000, options=options)
asrs_modelLS.train()

# %% [markdown]
# ## NTSB Dats Source

# %%
options={ "is_merge_taxonomy": True }
# 01. Get Data
ntsb_df = CoreUtils.get_data('ntsb', from_year=2000, to_year=2023)

ntsb_modelLS = ModelLS({'ntsb': ntsb_df}, 'ntsb', 10000, options=options)
ntsb_modelLS.train()

# %% [markdown]
# ## ASRS + NTSB 

# %%
options={ "is_merge_taxonomy": True }
# 01. Manual Labeling with specific sample
asrs_df = CoreUtils.get_data('asrs', from_year=2000, to_year=2023)

ntsb_df = CoreUtils.get_data('ntsb', from_year=2000, to_year=2023)

asrs_ntsb_modelLS = ModelLS({'asrs': asrs_df, 'ntsb': ntsb_df}, 'asrs_ntsb', 10000, options=options)
asrs_ntsb_modelLS.train()


