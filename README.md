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
### Setup .env
```
Rename .env.sample to .env
Change username and password in BASE_CONN and BASE_CONN_ASYNC
```
### Restore initial sql
```
mysql -u [username] -p < scripts/init_db.sql
```

### Re-Activate the VM
```
deactivate
```
```
Windows: python_modules\Scripts\activate
```
or
```
MacOS/Linux: source python_modules/bin/activate
```

## Start Service

### How to Run NoteBooks

#### Step 01: Run LS model
```
notebooks/notebook_main_LS.ipynb
```

#### Step 02: Run LSTM_ASRS model
```
notebooks/notebook_main_LSTM_ASRS.ipynb
```

#### Step 03: Run LSTM_NTSB model
```
notebooks/notebook_main_LSTM_NTSB.ipynb
```

#### Step 04: Run LSTM_ASRS_NTSB model
```
notebooks/notebook_main_LSTM_ASRS_NTSB.ipynb
```

### How to run Web Service

#### Step 01: Run web server
```
Step 02: uvicorn main:app --reload
```

#### Step 02: extract data For ASRS ds_id=1
```
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/extracts/1' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 03: extract data For NTSB ds_id=2
```
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/extracts/2' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 04: Load data For NTSB ds_id=1
```
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/loads/1' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 05: Load data For NTSB ds_id=2
```
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/loads/2' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 06: Run LS model Notebook
```
notebooks/notebook_main_LS.ipynb
```

#### Step 07: Go to Dashboard and simulate
```
http://0.0.0.0:8000/dashboard/simulator
```


## Freeze dependencies to requirements.txt
```
pip freeze > requirements.txt
```

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