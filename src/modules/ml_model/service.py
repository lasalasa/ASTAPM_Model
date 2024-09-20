import logging
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from src.core.model_lstm import LSTMModel
from src.core.model_ls import ModelLS

from src.core.pre_processor import DataPreprocessor
# from src.core.svm.svm import SVMModel

from . import models as MLModels
from . import schemas as MLSchemas
from ..data_source import models as DSModels

from fastapi import HTTPException, status
from sqlalchemy.future import select
from sqlalchemy import and_
from sqlalchemy import text
import pandas as pd
import numpy as np

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO or DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

async def do_train(
    db_session: Session,
    input_data: MLSchemas.ModelTrainSchema
):
    input_data_source = input_data.input_data_source
    input_model_name = input_data.input_model_name
    input_from_date = input_data.input_from_date
    input_to_date = input_data.input_to_date

    try:
        print("input_data_source=", input_data_source)
        result = await db_session.execute(select(DSModels.DataSource).where(DSModels.DataSource.ds_name == input_data_source))
        data_source = result.scalars().first()
        
        if data_source is None:
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")
        
        result = await db_session.execute(select(MLModels.MachineLearningModel).filter(MLModels.MachineLearningModel.ml_model_name == input_model_name))
        ml_model = result.scalars().first()
        
        if ml_model is None:
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ML model not found")

        input_data_source = input_data_source.lower()
        table_name = input_data_source
        narrative_table_name = f"{table_name}_narrative"
        safety_factors_table_name = f"{table_name}_safety_factors"

        query = text(f"""
            SELECT * FROM {table_name} AS a
            JOIN {narrative_table_name} AS b ON b.event_id = a.event_id
            JOIN {safety_factors_table_name} AS c ON c.event_id = a.event_id
            WHERE a.date BETWEEN :start_date AND :end_date
            ORDER BY a.date
        """)

        print("query=", query)

        # Execute the query with parameters
        result = await db_session.execute(query, {
            "start_date": input_from_date, #'2023-01-01 00:00:00',
            "end_date": input_to_date # '2023-01-31 23:59:59'
        })

        # Fetch all results, if needed
        rows = result.mappings().all()  # Fetch all rows as dictionaries

        if not rows:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

        df = pd.DataFrame(rows)

        print(input_model_name, "Sample Size=", df.shape)

        # return rows
        dfs = {}
        dfs[input_data_source] = df
        if input_model_name == "SVM":
            pass
        
        elif input_model_name == "LSTM_Predictor":
            # input_shape = (X_train.shape[1], 1)  # LSTM expects input in 3D shape (samples, timesteps, features)
            # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # model = LSTMModel(input_shape=input_shape)
            # model.train(X_train, y_train)
            # metrics = model.evaluate(X_test, y_test)
            # model.save("lstm_model.h5")

            model_LS = ModelLS(dfs, ds_name=input_data_source)
            model_LS.train()
            
            model_LSTM = LSTMModel(dfs, ds_name=input_data_source)
            model_LSTM.train()
            evaluation_report = model_LSTM.evaluate()
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        print(evaluation_report)

        return evaluation_report
        # return {"model_type": "LSTM_Predictor", "metrics": metrics}
    except Exception as e:
        logger.exception(f"An unexpected error occurred while training data source {input_data_source}: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": "Internal Server Error",
            "stackTrace": str(e)
        })
    

# def update_data_source(db: Session, ds_id: int, data_source: schemas.DataSourceUpdateSchema):
#     db_data_source = get_data_source(db, ds_id)
#     for key, value in data_source.model_dump().items():
#         setattr(db_data_source, key, value)
#     db.commit()
#     db.refresh(db_data_source)
#     return db_data_source

# def delete_data_source(db: Session, ds_id: int):
#     db_data_source = get_data_source(db, ds_id)
#     db.delete(db_data_source)
#     db.commit()
#     return db_data_source

# def re_collect_data_source(db: Session, ds_id: int):
#     data_source = db.query(models.DataSource).filter(models.DataSource.ds_id == ds_id).first()
#     if data_source is None:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")
    
#     return data_source

# def re_clean_data_source(db: Session, ds_id: int):
#     data_source = db.query(models.DataSource).filter(models.DataSource.ds_id == ds_id).first()
#     if data_source is None:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")
    
#     return data_source
