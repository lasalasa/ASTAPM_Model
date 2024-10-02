import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .modules.data_source.router import router as data_source_router
from .modules.ml_model.router import router as ml_model_router
from .database import session_manager

from src.config import settings

base_db_conn_async = settings.base_db_conn_async

logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# code adapted from (Lifespan Events - FastAPI, n.d.)
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not base_db_conn_async:
        raise Exception("Database URL is not configured correctly.")
    
    await session_manager.init("ASTAPM", base_db_conn_async)
    await session_manager.init("asrs_raw", base_db_conn_async)
    await session_manager.init("ntsb_raw", base_db_conn_async)

    yield
    await session_manager.close_all()
# code adapted end

app = FastAPI(lifespan=lifespan)

# code adapted from (Middleware - FastAPI, n.d.)
@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    try:
        async with session_manager.session("ASTAPM") as session:
            request.state.session = session
            response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Failed to handle session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
# code adapted end

app.include_router(data_source_router)
app.include_router(ml_model_router)