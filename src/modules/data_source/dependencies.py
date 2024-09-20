from fastapi import Header, HTTPException
from typing_extensions import Annotated
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from sqlalchemy.future import select

from . import models, schemas
from src.database import session_manager, get_db_session, get_child_db_session

async def get_token_header(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def get_query_token(token: str):
    if token != "jessica":
        raise HTTPException(status_code=400, detail="No Jessica token provided")
    
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