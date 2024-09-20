from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from . import schemas, service
# from ..database import get_db
from src.database import get_db_session

from .dependencies import get_token_header, get_data_source_db_session

router = APIRouter(
    prefix="/data-sources",
    tags=["Data Sources"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)

@router.post("/extracts/{ds_id}", 
    summary="Collect data from provided path",
    # response_model=schemas.DataSourceSchema
)
async def extracts(
    ds_id: int,
    db_session: AsyncSession = Depends(get_db_session),
    ds_and_ds_db_session: AsyncSession = Depends(get_data_source_db_session) 
):
    data_source, ds_db_session = ds_and_ds_db_session
    result = await service.extract_data(db_session, ds_db_session, data_source, ds_id)
    return result

@router.post("/loads/{ds_id}",
    summary="Clean and restore the data source data")
async def re_clean_data_source(
    ds_id: int,
    db_session: AsyncSession = Depends(get_db_session),
    ds_and_ds_db_session: AsyncSession = Depends(get_data_source_db_session) 
):
    data_source, ds_db_session = ds_and_ds_db_session
    result = await service.load_data(db_session, ds_db_session, data_source, ds_id)
    return result

@router.get("/trends-overview",
    summary="Clean and restore the data source data")
async def re_clean_data_source(
    ds_id: int,
    db_session: AsyncSession = Depends(get_db_session),
    ds_and_ds_db_session: AsyncSession = Depends(get_data_source_db_session) 
):
    data_source, ds_db_session = ds_and_ds_db_session
    result = await service.load_data(db_session, ds_db_session, data_source, ds_id)
    return result