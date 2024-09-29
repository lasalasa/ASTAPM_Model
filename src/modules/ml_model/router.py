from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from . import schemas, service
from src.database import get_db_session

router = APIRouter(
    prefix="/ml-models",
    tags=["Model Management"],
    responses={404: {"description": "Not found"}},
)

@router.post("/train",
    summary="Train data and save model",
    status_code=status.HTTP_201_CREATED)
async def train_ml_model(
    input_data: schemas.ModelTrainSchema,
    db_session: AsyncSession = Depends(get_db_session)
):
    return await service.do_train(db_session, input_data)

@router.post("/predict",
    summary="Collect data from provided path",
    status_code=status.HTTP_201_CREATED)
async def train_ml_model(
    model_train_data: schemas.ModelTrainSchema,
    db_session: AsyncSession = Depends(get_db_session)
):
    return await service.train_ml_model(db_session, model_train_data)

@router.post("/evaluate", 
    summary="Collect data from provided path",
    status_code=status.HTTP_200_OK)
async def create_data_source(ds_id: int, db_session: AsyncSession = Depends(get_db_session)):
    return service.re_collect_data_source(db_session, ds_id)