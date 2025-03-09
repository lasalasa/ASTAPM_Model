# ASTAPM (Aviation Safety Trends Analysis and Predictive Model)

## Setup Environment

### 1. Create a Virtual Environment (VM)
```sh
py -3.11 -m venv python_modules
```

### 2. Activate the Virtual Environment
#### Windows:
```sh
python_modules\Scripts\activate
```
#### macOS/Linux:
```sh
source python_modules/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Setup Environment Variables
- Rename `.env.sample` to `.env`.
- Update `BASE_CONN` and `BASE_CONN_ASYNC` with the correct database credentials.

### 5. Restore Initial SQL Database
```sh
mysql -u [username] -p < scripts/init_db.sql
```

### 6. Reactivate Virtual Environment (if needed)
To deactivate and reactivate:
```sh
deactivate
```
#### Windows:
```sh
python_modules\Scripts\activate
```
#### macOS/Linux:
```sh
source python_modules/bin/activate
```

---

## Start Service

### Running Notebooks

#### Step 1: Run LS Model
```sh
notebooks/notebook_main_LS.ipynb
```

#### Step 2: Run LSTM_ASRS Model
```sh
notebooks/notebook_main_LSTM_ASRS.ipynb
```

#### Step 3: Run LSTM_NTSB Model
```sh
notebooks/notebook_main_LSTM_NTSB.ipynb
```

#### Step 4: Run LSTM_ASRS_NTSB Model
```sh
notebooks/notebook_main_LSTM_ASRS_NTSB.ipynb
```

### Running Web Service

#### Step 1: Start the Web Server
```sh
uvicorn main:app --reload
```

#### Step 2: Extract Data for ASRS (ds_id=1)
```sh
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/extracts/1' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 3: Extract Data for NTSB (ds_id=2)
```sh
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/extracts/2' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 4: Load Data for ASRS (ds_id=1)
```sh
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/loads/1' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 5: Load Data for NTSB (ds_id=2)
```sh
curl -X 'POST' \
  'http://0.0.0.0:8000/data-sources/loads/2' \
  -H 'accept: application/json' \
  -d ''
```

#### Step 6: Run LS Model Notebook
```sh
notebooks/notebook_main_LS.ipynb
```

#### Step 7: Access Dashboard and Simulate
```sh
http://0.0.0.0:8000/dashboard/simulator
```

---

## Freeze Dependencies
To save installed dependencies into `requirements.txt`:
```sh
pip freeze > requirements.txt
```

---

## Project Structure
```
ASTAPM/
│
├── data/
│   ├── local_ex/
│   └── local_im/
│
├── src/
│   ├── core/
│   ├── extensions/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── models.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   ├── service.py
│   ├── data_source/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── models.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   ├── service.py
│   ├── __init__.py
│   ├── config.py
│   ├── constant.py
│   ├── database.py
│   ├── app.py
│
├── tests/
├── notebooks/
├── web/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Notes
- Ensure the `.env` file is correctly set up before running the application.
- MySQL database must be running and accessible.
- The API server runs on `http://0.0.0.0:8000` by default.
- The system processes both ASRS and NTSB data for aviation safety trend analysis.

