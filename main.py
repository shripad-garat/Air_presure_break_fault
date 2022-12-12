from sensor.logger import logging
from sensor.exception import SensorException
from sensor.pipeline import traning_pipeline
from sensor.pipeline import batch_predection
import os,sys
INPUT_FILE_PATH = path = os.path.join("D:\FSDS PROJECT\Project2_\Predict.csv")



if __name__=="__main__":
    try:
        traning_pipeline.start_traning_pipeline()
        print(batch_predection.BatchPrediction(INPUT_FILE_PATH))

    except Exception as e:
        raise SensorException(e,sys)