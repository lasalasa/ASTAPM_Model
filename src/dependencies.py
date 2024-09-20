from fastapi import Header, HTTPException
from typing_extensions import Annotated

from .database import get_db_session
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession


async def get_token_header(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def get_query_token(token: str):
    if token != "jessica":
        raise HTTPException(status_code=400, detail="No Jessica token provided")
    

DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]