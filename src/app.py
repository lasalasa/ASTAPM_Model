import uvicorn
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sqlalchemy.orm import Session

from .dependencies import get_query_token, get_token_header
from .modules.data_source.router import router as data_source_router
from .modules.ml_model.router import router as ml_model_router
from .database import session_manager

from src.config import settings

# from web.app import app as Dashboard

base_db_conn_async = settings.base_db_conn_async

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Example configuration check
    if not base_db_conn_async:
        raise Exception("Database URL is not configured correctly.")
    
    await session_manager.init("ASTAPM", base_db_conn_async)
    await session_manager.init("asrs_raw", base_db_conn_async)
    await session_manager.init("ntsb_raw", base_db_conn_async)

    yield
    await session_manager.close_all()

# app = FastAPI(lifespan=lifespan, dependencies=[Depends(get_query_token)])
app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    try:
        # Create async session from session manager
        async with session_manager.session("ASTAPM") as session:
            request.state.session = session
            response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Failed to handle session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

app.include_router(data_source_router)
app.include_router(ml_model_router)

# Bigger Applications - Multiple Files
# https://fastapi.tiangolo.com/tutorial/bigger-applications/#import-fastapi
# https://medium.com/@amirm.lavasani/how-to-structure-your-fastapi-projects-0219a6600a8f
# https://github.com/zhanymkanov/fastapi-best-practices?source=post_page-----0219a6600a8f--------------------------------#1-project-structure-consistent--predictable

# Tesla Stock Forecasting 📈 Multi-Step Stacked LSTM
# https://www.kaggle.com/code/guslovesmath/tesla-stock-forecasting-multi-step-stacked-lstm

# https://fastapi.tiangolo.com/tutorial/sql-databases/#review-all-the-files