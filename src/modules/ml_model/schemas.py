from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

# code adapted from (SQL (Relational) Databases - FastAPI, n.d.)
class ModelType(str, Enum):
    SVM = "SVM"
    LSTM = "LSTM"

class FeatureEngineeringType(str, Enum):
    NMF = "NMF"
    ADSGL = "ADSGL"

class ModelBaseSchema(BaseModel):
    ml_model_name: str
    ml_model_type: ModelType
    ml_model_description: Optional[str] = None
    feature_engineering: FeatureEngineeringType

class ModelCreateSchema(ModelBaseSchema):
    pass

class ModelUpdateSchema(ModelBaseSchema):
    pass

# Sample Request
'''
{
    "input_data_source": "asrs",
    "input_model_name": "LSTM_Predictor",
    "input_from_date": "2023-01-01 00:00:00",
    "input_to_date": "2023-01-31 23:59:59"
}
'''
class ModelTrainSchema(BaseModel):
    input_data_source: str
    # input_sample_size: int
    input_model_name: str
    input_from_date: str
    input_to_date: str
    input_model_ls_version: int

    # feature_engineering: FeatureEngineeringType

class ModelSchema(ModelBaseSchema):
    ml_model_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TrainModelSchema(BaseModel):
    train_model_id: int
    ml_model_id: int
    training_data: Dict[str, List]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
# code adapted end
