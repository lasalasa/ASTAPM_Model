# ASTAPM (Aviation Safety Trends Analysis and Predictive Model)

## Set-up Environment

### Create a VM
```
python3 -m venv python_modules
```

### Activate the VM (Windows)
```
python_modules\Scripts\activate
```

### Activate the VM (macOS/Linux)
```
source python_modules/bin/activate
```
### Install dependencies
```
pip install -r requirements.txt
```
### Rename .env.sample to .env

### Freeze dependencies to requirements.txt
```
pip freeze > requirements.txt
pip freeze > ./web/requirements.txt
```

### Deactivate the VM
```
deactivate
```

## Start Service

### How to Run NoteBooks
```
Step 01: Run => notebook_main_LS.ipynb
Step 02: Run => notebook_main_LSTM_ASRS.ipynb
Step 03: Run => notebook_main_LSTM_NTSB.ipynb
Step 04: Run => notebook_main_LSTM_ASRS_NTSB.ipynb
```

### How to run Web Service

uvicorn main:app --reload


## Project Structure

```
ASTAPM/
│
├── data/
│   ├── local_ex/
│   └── local_im/
│
├── src/
│   └── core
│   └── extensions
│   └── modules
│   │   └── modules
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py
│   │   │   ├── exceptions.py
│   │   │   ├── models.py
│   │   │   ├── router.py
│   │   │   ├── schemas.py
│   │   │   ├── service.py
│   │   └── data_source
│   │   │   ├── __init__.py
│   │   │   ├── dependencies.py
│   │   │   ├── exceptions.py
│   │   │   ├── models.py
│   │   │   ├── router.py
│   │   │   ├── schemas.py
│   │   │   ├── service.py
│   ├── __init__.py
│   ├── config.py
│   ├── constant.py
│   ├── database.py
│   ├── app.py
│
├── tests
│
├── notebooks
├── web
│
├── .gitignore
├── README.md
└── requirements.txt
```