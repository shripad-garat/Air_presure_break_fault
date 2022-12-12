from sensor.entity import artifact_entity,config_entity
from sensor.logger import logging
from sensor.exception import SensorException
from sensor import utility
from sensor.predector import ModelResolver
import numpy as np
import pandas as pd
import os,sys

PREDICTION_DIR = "prediction"

 
def BatchPrediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        model_solver = ModelResolver()
        logging.info(f"Starting the Batch Predection")
        df = pd.read_csv(input_file_path)
        df.replace("na",np.NAN,inplace=True)
        logging.info("Loading the Input file for predection")
        logging.info("Importing the model, transformer and target encoder")
        model_path = model_solver.get_latest_model_path()
        transformer_path = model_solver.get_latest_transformer_path()
        target_encoder_path = model_solver.get_latest_target_encoder_path()
        model = utility.load_object(model_path)
        transformer = utility.load_object(transformer_path)
        target_encoder = utility.load_object(target_encoder_path)
        logging.info("Transforming the data")

        transformed_df = transformer.transform(df)
        logging.info("Making the batch predection on given input data")
        y_predict = model.predict(transformed_df)
        y_predict_cat = target_encoder.inverse_transform(y_predict)
        logging.info("Saving the prewdection in the csv format")
        df["Predicted_Value"] = y_predict
        df['Predicted_Categorical'] = y_predict_cat
        file_name = os.path.basename(input_file_path)
        prediction_dir_path = os.path.join(PREDICTION_DIR,file_name)
        df.to_csv(prediction_dir_path,header=True,index=False)
        logging.info("Returining the path for the predection filess")
        return prediction_dir_path

    except Exception as e:
        raise SensorException(e,sys)