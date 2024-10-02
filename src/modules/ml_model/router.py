from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from . import schemas, service
from src.database import get_db_session

# code adapted from (SQL (Relational) Databases - FastAPI, n.d.)
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
# code adapted end