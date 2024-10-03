# %% [markdown]
# # Human Factors Prediction Model => LSTM(ASRS)

# %% [markdown]
# ## Import Notebook

# %%
%run notebook_model_ls.ipynb
%run notebook_model_lstm.ipynb

# %%
from sklearn.metrics import f1_score, accuracy_score, hamming_loss

def show_accuracy(model):

    y_true = model.Y_test
    y_pred = model.Y_pred

    accuracy = accuracy_score(y_true, y_pred)

    # code adapted from (F1_Score, n.d.)
    macro_f1_score = f1_score(y_true, y_pred, average='macro')
    micro_f1_score = f1_score(y_true, y_pred, average='micro')
    weighted_f1_score = f1_score(y_true, y_pred, average='weighted')
    # end of adapted code

    hamming_loss_value = hamming_loss(y_true, y_pred)

    print(f'Accuracy: {(accuracy*100):.2f}')
    print(f'Micro F1 Score: {(micro_f1_score*100):.2f}')
    print(f'Macro F1 Score: {(macro_f1_score*100):.2f}')
    print(f'Weighted F1 Score: {(weighted_f1_score*100):.2f}')
    print(f'Hamming Loss: {hamming_loss_value:.4f}')

def show_report(model):
    print(model.classification_report)

# %%
# 01. Get Data
asrs_df = CoreUtils.get_data('asrs', from_year=2023, to_year=2023)
# asrs_df.sort_values(by='date', ascending=False, inplace=True)
# asrs_df = asrs_df.head(500)

ls_version = 2 # Default version=2
ls_name = 'asrs_ntsb'
ds_name = 'asrs'

# %%
def call_modal(options):
    dfs = { ds_name: asrs_df.copy() }
    model = LSTMModel(dfs, ds_name=ds_name, options=options)
    model.train()

    # 03. Evaluate Model
    evaluate_result = model.evaluate()
    return model

# %% [markdown]
# ## LSTM(ASRS) + LS(Hybrid)

# %%
options = {
    "sample_size": 0, 
    "max_length": 200, 
    "max_nb_words": 20000, 
    "is_enable_smote": False,
    "is_enable_asasyn": False,
    "is_enable_class_weight": False,
    "ls_name": ls_name,
    "ls_version": ls_version
}

lstm_model = call_modal(options)

# %% [markdown]
# ## LSTM(ASRS) + LS(Hybrid) + Class Weight

# %%
options = {
    "sample_size": 0, 
    "max_length": 200, 
    "max_nb_words": 20000, 
    "is_enable_smote": False,
    "is_enable_asasyn": False,
    "is_enable_class_weight": True,
    "ls_name": ls_name,
    "ls_version": ls_version
}

lstm_model_weight = call_modal(options)

# %% [markdown]
# ## LSTM(ASRS) + LS(Hybrid) + SMOTE

# %%
options = {
    "sample_size": 0, 
    "max_length": 200, 
    "max_nb_words": 20000, 
    "is_enable_smote": True,
    "is_enable_asasyn": False,
    "is_enable_class_weight": False,
    "ls_name": ls_name,
    "ls_version": ls_version
}

lstm_model_smote = call_modal(options)

# %% [markdown]
# ## LSTM(ASRS)+LS(Hybrid)+ADASYN

# %%
options = {
    "sample_size": 0, 
    "max_length": 200, 
    "max_nb_words": 20000, 
    "is_enable_smote": False,
    "is_enable_asasyn": True,
    "is_enable_class_weight": False,
    "ls_name": ls_name,
    "ls_version": ls_version
}

# 02. Train Model
lstm_model_asasyn = call_modal(options)

# %% [markdown]
# ## Summary of Accuracy

# %%
# Show Accuracy
print('ASRS+LS(Hybrid)=========')
show_accuracy(lstm_model)
print('ASRS+LS(Hybrid)+Class Imbalance=========')
show_accuracy(lstm_model_weight)
print('ASRS+LS(Hybrid)+SMOTE=========')
show_accuracy(lstm_model_smote)
print('ASRS+LS(Hybrid)+ASAYN=========')
show_accuracy(lstm_model_asasyn)

# %%
# Show Accuracy
print('ASRS+LS(Hybrid)=========')
show_report(lstm_model)

# %%
print('ASRS+LS(Hybrid)+Class Imbalance=========')
show_report(lstm_model_weight)

# %%
print('ASRS+LS(Hybrid)+SMOTE=========')
show_report(lstm_model_smote)

# %%
print('ASRS+LS(Hybrid)+ASAYN=========')
show_report(lstm_model_asasyn)

# %% [markdown]
# ## References
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html


