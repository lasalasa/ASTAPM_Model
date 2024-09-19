# ASTAPM (Aviation Safety Trends Analysis and Predictive Model)

## Set up Virtual Environment(VM)

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
pip install numpy pandas
```

### Freeze dependencies to requirements.txt
```
pip freeze > requirements.txt
```

### Deactivate the VM
```
deactivate
```

## Run Service

DEV: `fastapi dev main.py`

## Project Structure

```
ASTAPM/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
│
├── src/
│   └── data_source
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── models.py
│   │   ├── router.py
│   │   ├── schemas.py
│   │   ├── service.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── config.py
│   ├── constant.py
│   ├── database.py
│   ├── dependencies.py
│   ├── exceptions.py  # global exceptions
│   ├── logger.py
│   ├── main.py
│   ├── models.py # global models
│   ├── pagination.py  # global module e.g. pagination
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   └── test_file_operations.py
│
├── notebooks/
│   ├── analysis.ipynb
│   └── exploration.ipynb
│
├── docs/
│   └── README.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Reference
https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
https://medium.com/@joshuale/a-practical-guide-to-python-project-structure-and-packaging-90c7f7a04f95

FastAPI: https://fastapi.tiangolo.com/#example


https://andreantonacci.github.io/aviation-accidents/