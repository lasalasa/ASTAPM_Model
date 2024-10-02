# Outer service module
import logging
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from sqlalchemy.future import select
from sqlalchemy import text
from fastapi import HTTPException, status

# Inner Service module
from src.core.model_lstm import LSTMModel
from . import models as MLModels
from . import schemas as MLSchemas
from ..data_source import models as DSModels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
        
        target_data_source = input_data_source.lower()

        print(f'target_data_source={target_data_source}')

        dfs = {}
        dfs_list = []
        for ds_name_item in input_data_source_list:

            result = await db_session.execute(select(DSModels.DataSource).where(DSModels.DataSource.ds_name == ds_name_item))
            data_source = result.scalars().first()
            
            if data_source is None:
                return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found")

            # table_name = input_data_source
            # narrative_table_name = f"{table_name}_narrative"
            # safety_factors_table_name = f"{table_name}_safety_factors"

            # Testing purpose
            if ds_name_item.lower() == "asrs":
                input_from_date = '2023-01-01 00:00:00'
                input_to_date = '2023-06-31 23:59:59'
                max_length = 150
                max_nb_words = 10000
                is_enable_smote = False
            elif ds_name_item.lower() == "ntsb":
                input_from_date = '2020-01-01 00:00:00'
                input_to_date = '2023-01-31 23:59:59'
                max_length = 200
                max_nb_words = 10000,
                is_enable_smote = False
            

            query = text(f"""
                SELECT * FROM {ds_name_item}
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date
            """)
            print("query=", query, input_from_date, input_to_date)

            # Execute the query with parameters
            result = await db_session.execute(query, {
                "start_date": input_from_date, #'2023-01-01 00:00:00',
                "end_date": input_to_date # '2023-01-31 23:59:59'
            })

            # Fetch all results, if needed
            rows = result.mappings().all()

            if not rows:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Data not found")

            df_item = pd.DataFrame(rows)
            dfs[ds_name_item] = df_item
            dfs_list.append(df_item)
            print(input_model_name, "Sample Size=", df_item.shape)


        dfs[input_data_source] = pd.concat(dfs_list, axis=0).reset_index(drop=True)

        # Testing purpose
        if target_data_source == "asrs_ntsb":
            # input_from_date = '2023-01-01 00:00:00'
            # input_to_date = '2023-12-31 23:59:59'
            max_length = 200
            max_nb_words = 7500,
            is_enable_smote = False

        if input_model_name == "LSTM_Predictor":
            print(max_length, max_nb_words)

            options = {
                "sample_size": 0, 
                "max_length": max_length, 
                "max_nb_words": max_nb_words, 
                "is_enable_smote": is_enable_smote,
                "is_enable_asasyn": False,
                "is_enable_class_weight": False,
                "ls_name": 'asrs_ntsb',
                "ls_version": input_model_ls_version
            }
            model_LSTM = LSTMModel(dfs, ds_name=input_data_source, options=options)
            model_LSTM.train()
            evaluation_report = model_LSTM.evaluate()
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
        # print(evaluation_report)

        return evaluation_report
    except Exception as e:
        logger.exception(f"An unexpected error occurred while training data source {input_data_source}: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": "Internal Server Error",
            "stackTrace": str(e)
        })
