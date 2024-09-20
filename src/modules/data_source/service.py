from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sqlalchemy import text, insert
import pandas as pd

from fastapi import HTTPException, status

import logging

from . import models, schemas
from src.core.etl.asrs_etl import AsrsEtl
from src.core.etl.ntsb_etl import NtsbEtl
from src.extensions.pd_extension import PdExtension
from src.database import session_manager, get_db_session, get_child_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO or DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def get_data_source(db: Session, ds_id: int):
    data_source = db.query(models.DataSource).filter(models.DataSource.ds_id == ds_id).first()
    if data_source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")
    return data_source

def get_data_sources(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.DataSource).offset(skip).limit(limit).all()

def create_data_source(db: Session, data_source: schemas.DataSourceCreateSchema):
    db_data_source = models.DataSource(**data_source.model_dump())
    db.add(db_data_source)
    db.commit()
    db.refresh(db_data_source)
    return db_data_source

def update_data_source(db: Session, ds_id: int, data_source: schemas.DataSourceUpdateSchema):
    db_data_source = get_data_source(db, ds_id)
    for key, value in data_source.model_dump().items():
        setattr(db_data_source, key, value)
    db.commit()
    db.refresh(db_data_source)
    return db_data_source

def delete_data_source(db: Session, ds_id: int):
    db_data_source = get_data_source(db, ds_id)
    db.delete(db_data_source)
    db.commit()
    return db_data_source

async def extract_data(db_session: Session, ds_db_session: Session, data_source: models.DataSource, ds_id: int):
    
    try:
        if data_source.ds_type == "csv":
            print("csv file format")

            csv_etl = AsrsEtl(data_source)
            await csv_etl.extract_data()
            #SQL raw query
            query = text("SELECT * FROM Events LIMIT 500")

        elif data_source.ds_type == "mdb":
            print("mdb file format")

            mdb_etl = NtsbEtl(data_source)
            await mdb_etl.extract_data()
            #SQL raw query
            query = text("SELECT * FROM events LIMIT 500")

        # Execute the query
        result = await ds_db_session.execute(query)

        # Fetch the first result as a dictionary
        data = result.mappings().first() 

        if not data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")

        print("Successfully extracted")

        return dict(data)
    except Exception as e:
        logger.exception(f"An unexpected error occurred while processing data source {ds_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

async def load_data(db_session: Session, ds_db_session: Session, data_source: models.DataSource, ds_id: int):
    try:

        if data_source.ds_type == "csv":
            csv_etl = AsrsEtl(data_source)
            await csv_etl.transform_data()
            await csv_etl.load_data()
        elif data_source.ds_type == "mdb":
            mdb_etl = NtsbEtl(data_source)
            await mdb_etl.transform_data()
            await mdb_etl.load_data()

        print("Successfully loaded")

        return {"message": "Data cleaned and saved successfully"}
        
    except Exception as e:
        logger.exception(f"An unexpected error occurred while cleaning data {ds_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
