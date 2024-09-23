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
    input_model_ls_version = input_data.input_model_ls_version

    try:
        print("input_data_source=", input_data_source)

        input_data_source_list = input_data_source.split('_')

        print("data_source_list=",input_data_source_list)

        result = await db_session.execute(select(MLModels.MachineLearningModel).filter(MLModels.MachineLearningModel.ml_model_name == input_model_name))
        ml_model = result.scalars().first()
        
        if ml_model is None:
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ML model not found")

        dfs = {}
        dfs_list = []
        for ds_name_item in input_data_source_list:

            result = await db_session.execute(select(DSModels.DataSource).where(DSModels.DataSource.ds_name == ds_name_item))
            data_source = result.scalars().first()
            
            if data_source is None:
                return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")

            input_data_source = input_data_source.lower()
            table_name = input_data_source
            # narrative_table_name = f"{table_name}_narrative"
            # safety_factors_table_name = f"{table_name}_safety_factors"

            query = text(f"""
                SELECT * FROM {ds_name_item}
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date
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

            df_item = pd.DataFrame(rows)
            dfs[ds_name_item] = df_item
            dfs_list.append(df_item)
            print(input_model_name, "Sample Size=", df_item.shape)


        dfs[input_data_source] = pd.concat(dfs_list, axis=0).reset_index(drop=True)

        # query = text(f"""
        #     SELECT * FROM {table_name}
        #     WHERE date BETWEEN :start_date AND :end_date
        #     ORDER BY date
        # """)

        # print("query=", query)

        # # Execute the query with parameters
        # result = await db_session.execute(query, {
        #     "start_date": input_from_date, #'2023-01-01 00:00:00',
        #     "end_date": input_to_date # '2023-01-31 23:59:59'
        # })

        # # Fetch all results, if needed
        # rows = result.mappings().all()  # Fetch all rows as dictionaries

        # if not rows:
        #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

        # df = pd.DataFrame(rows)

        # print(input_model_name, "Sample Size=", df.shape)

        # return rows
        # dfs = {}
        # dfs[input_data_source] = df

        if input_model_name == "SVM":
            pass
        
        elif input_model_name == "LSTM_Predictor":
            
            model_LSTM = LSTMModel(dfs, ds_name=input_data_source, ls_version=input_model_ls_version, sample_size=0)
            model_LSTM.train()
            evaluation_report = model_LSTM.evaluate()
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        # print(evaluation_report)

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
