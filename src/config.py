import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from a .env file
load_dotenv()

# Extract environment variables with defaults and error handling
main_db_name = os.getenv('DB_NAME')
base_db_conn = os.getenv('BASE_CONN')
base_db_conn_async = os.getenv('BASE_CONN_ASYNC')

if not main_db_name or not base_db_conn or not base_db_conn_async:
    raise ValueError("Environment variables 'BASE_CONN', BASE_CONN_ASYNC and DB_NAME must be set.")

# code adapted from ThomasAitken (n.d.)
class Settings(BaseSettings):
    main_db_name: str = main_db_name
    base_db_conn: str = base_db_conn
    base_db_conn_async: str = base_db_conn_async
    ex_folder_path: str = "data/local_ex/astapm"
    echo_sql: bool = True
    test: bool = False
    project_name: str = "My FastAPI project"
    token_secret: str = "my_dev_secret"
    log_level: str = "DEBUG"

settings = Settings()  # type: ignore
# end of adapted code
# TODO REF=https://github.com/ThomasAitken/demo-fastapi-async-sqlalchemy/blob/main/backend/app/config.py
# TODO REF=https://docs.pydantic.dev/latest/concepts/pydantic_settings/