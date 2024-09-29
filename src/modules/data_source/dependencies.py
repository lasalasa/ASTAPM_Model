from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from sqlalchemy.future import select

from . import models
from src.database import get_db_session, get_child_db_session
    
async def get_data_source_db_session(ds_id: int, db_session: AsyncSession = Depends(get_db_session)):
    print("*************BBBBB")
    result = await db_session.execute(select(models.DataSource).where(models.DataSource.ds_id == ds_id))
    data_source = result.scalars().first()
    
    if not data_source:
        raise HTTPException(status_code=404, detail=f"DataSource with id {ds_id} not found")

    db_name = data_source.ds_raw_db
    
    # Get and return the session for the child database
    child_db_session = await get_child_db_session(db_name)
    return data_source, child_db_session