import logging

from typing import Dict, AsyncIterator
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession, AsyncConnection
from sqlalchemy.orm import sessionmaker as async_sessionmaker
import contextlib
from sqlalchemy.orm import declarative_base
from fastapi import Request
from sqlalchemy import text, MetaData

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# code adapted from Caminha (2023)
class DatabaseSessionManager:
    def __init__(self):
        self._engines: Dict[str, AsyncEngine] = {}
        self._sessionmakers: Dict[str, async_sessionmaker] = {}
        self._metadata: Dict[str, MetaData] = {} 

    async def __validate_database(self, db_name: str, host: str):
        logger.info(f"Validating existence of database: {db_name}")
        engine = create_async_engine(host)

        async with engine.connect() as conn:
            # code adapted from (How to Check if Mysql Database Exists, n.d.)
            query = text(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = :db_name")
            # end of adapted code

            result = await conn.execute(query, {"db_name": db_name})
            db_exists = result.scalar() is not None

            if not db_exists:
                logger.info(f"Database {db_name} does not exist. Creating...")
                await conn.execute(text(f"CREATE DATABASE {db_name}"))
            else:
                logger.info(f"Database {db_name} already exists.")

    async def init(self, db_name: str, host: str):
        if db_name in self._engines:
            raise Exception(f"Database {db_name} is already initialized.")
        
        await self.__validate_database(db_name, host)

        print(f"{host}/{db_name}")
        engine = create_async_engine(f"{host}/{db_name}")

        # Create a sessionmaker bound to the engine
        session_maker = async_sessionmaker(
            bind=engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        self._engines[db_name] = engine
        self._sessionmakers[db_name] = session_maker

        logger.info(f"Engine and sessionmaker for {db_name} initialized.")

    async def get_metadata(self, db_name: str) -> MetaData:
        if db_name not in self._engines:
            raise Exception(f"Database {db_name} is not initialized.")

        if db_name not in self._metadata:
            # Create a new MetaData object and reflect the database schema
            logger.info(f"Reflecting metadata for database: {db_name}")
            metadata = MetaData()
            async with self._engines[db_name].connect() as connection:
                await connection.run_sync(metadata.reflect)
            self._metadata[db_name] = metadata
        
        return self._metadata[db_name]
    
    async def close(self, db_name: str):
        if db_name not in self._engines:
            raise Exception(f"Database {db_name} is not initialized.")
        
        await self._engines[db_name].dispose()
        del self._engines[db_name]
        del self._sessionmakers[db_name]
        logger.info(f"Closed and disposed engine for {db_name}")

    async def close_all(self):
        for engine in self._engines.values():
            await engine.dispose()
            # logger.info(f"Closed and disposed engine for {db_name}")
        self._engines.clear()
        self._sessionmakers.clear()
        logger.info("All engines closed and disposed.")

    @contextlib.asynccontextmanager
    async def connect(self, db_name: str) -> AsyncIterator[AsyncConnection]:
        if db_name not in self._engines:
            raise Exception(f"Database {db_name} is not initialized.")
        
        async with self._engines[db_name].begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self, db_name: str) -> AsyncIterator[AsyncSession]:
        if db_name not in self._sessionmakers:
            raise Exception(f"Database {db_name} is not initialized.")
        
        session = self._sessionmakers[db_name]()
        if session is None:
            logger.error("Failed to create a session. The session object is None.")
            print("Failed to create a session. The session object is None.")
            raise Exception("Failed to create a session. The session object is None.")
        
        try:
            yield session
        except Exception as e:
            logger.error(f"Error during session with {db_name}: {e}")
            await session.rollback()
            raise e
        finally:
            await session.close()
            logger.info(f"Closed session for {db_name}")

    # Used for testing
    async def create_all(self, db_name: str, connection: AsyncConnection):
        if db_name not in self._engines:
            raise Exception(f"Database {db_name} is not initialized.")
        await connection.run_sync(Base.metadata.create_all)

    async def drop_all(self, db_name: str, connection: AsyncConnection):
        if db_name not in self._engines:
            raise Exception(f"Database {db_name} is not initialized.")
        await connection.run_sync(Base.metadata.drop_all)

session_manager = DatabaseSessionManager()

async def get_db_session(request: Request) -> AsyncSession:
    return request.state.session

async def get_child_db_session(db_name: str) -> AsyncSession:
    async with session_manager.session(db_name) as session:
        return session
# end of adapted code

