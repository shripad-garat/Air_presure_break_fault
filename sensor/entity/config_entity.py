import os ,sys
from datetime import datetime
from sensor.logger import logging
from sensor.exception import SensorException
FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

class TraningPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise SensorException(error_message=e,error_detail=sys)
    

class DataIngestionConfig:
    def __init__(self,traning_pipeline_config:TraningPipelineConfig):
        try:
            self.Database_name = "AirPresureFault"
            self.collection_name = "sensor"
            self.data_injestion_dir = os.path.join(traning_pipeline_config.artifact_dir,"data_injestion")
            self.featture_store_file_path = os.path.join(self.data_injestion_dir,"featureStore",FILE_NAME)
            self.train_file_path = os.path.join(self.data_injestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_injestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise SensorException(error_message=e,error_detail=sys)

    def to_dict(self):
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(error_message=e,error_detail=sys)

class DataValidationConfig:
    
    
    def __init__(self,traning_pipeline_config:TraningPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(traning_pipeline_config.artifact_dir,"data_validation")
            self.report_file_path = os.path.join(self.data_validation_dir, "report.yaml")
            self.missing_threshold:float = 0.2
            self.base_file_path = os.path.join("D:\FSDS PROJECT\Project2_\data.csv")

        except Exception as e:
            raise SensorException(e,sys)





class DataTransformationConfig:
    def __init__(self,traning_pipeline_config:TraningPipelineConfig):
        try: 
            self.data_transformation_dir = os.path.join(traning_pipeline_config.artifact_dir,"data_transformation")
            self.transformer_object_path = os.path.join(self.data_transformation_dir,'transformer',TRANSFORMER_OBJECT_FILE_NAME)
            self.transformed_train_path = os.path.join(self.data_transformation_dir,"transformed_train_file",TRAIN_FILE_NAME.replace("csv","npz"))
            self.transformed_test_path = os.path.join(self.data_transformation_dir,"transformed_test_file",TEST_FILE_NAME.replace("csv","npz"))
            self.target_encoder_path = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)

        
        except Exception as e:
            raise SensorException(e,sys)


class ModelTraningConfig:

    def __init__(self,traning_pipeline_config:TraningPipelineConfig):
        try:
            self.model_traning_dir = os.path.join(traning_pipeline_config.artifact_dir,"Model_traning")
            self.model_path = os.path.join(self.model_traning_dir,"model",MODEL_FILE_NAME)
            self.expected_score = 0.7
            self.overfitting_threshold = 0.1

        except Exception as e:
            raise SensorException(e,sys)


class ModelEvaluationConfig:
    def __init__(self,traning_pipeline_config:TraningPipelineConfig):
        try:
            self.xhange_threshold = 0.01

        except Exception as e:
            raise SensorException(e,sys)


class ModelPusherConfig:

    def __init__(self,traning_pipeline_config:TraningPipelineConfig):
        try:
            self.model_pusher_dir = os.path.join(traning_pipeline_config.artifact_dir,"model_pusher")
            self.save_model_dir = os.path.join("saved_models")
            self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
            self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
            self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)
            self.pusher_target_encode_path = os.path.join(self.pusher_model_dir,TARGET_ENCODER_OBJECT_FILE_NAME)


        except Exception as e:
            raise SensorException(e,sys)