from sensor.components import data_ingestion
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.modle_traning import ModelTraning
from sensor.components.modle_evaluation import ModelEvaluation
from sensor.components.modle_pusher import ModelPusher
import os,sys
from sensor.entity import config_entity

def start_traning_pipeline():
    try:
        traning_pipeline_config = config_entity.TraningPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(traning_pipeline_config=traning_pipeline_config)
        Data_ingestion = data_ingestion.DataInjestion(data_injestion_config=data_ingestion_config)
        data_ingestion_artifact = Data_ingestion.initiate_data_injection()
        data_validation_config = config_entity.DataValidationConfig(traning_pipeline_config=traning_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()
        data_transformation_config = config_entity.DataTransformationConfig(traning_pipeline_config=traning_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        model_tarning_config = config_entity.ModelTraningConfig(traning_pipeline_config= traning_pipeline_config)
        model_traning = ModelTraning(model_tarning_config,data_transformation_artifact)
        model_tarning_artifact = model_traning.inidated_model_traning()
        model_eval_config = config_entity.ModelEvaluationConfig(traning_pipeline_config= traning_pipeline_config)
        model_eval = ModelEvaluation(model_evaluation_config=model_eval_config,model_traning_artifact=model_tarning_artifact,data_transformation_artifact=data_transformation_artifact,data_ingestion_artifat=data_ingestion_artifact)
        model_eval_artifact = model_eval.initated_model_evaluation()
        model_pusher_config = config_entity.ModelPusherConfig(traning_pipeline_config=traning_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,model_traning_artifact=model_tarning_artifact,data_transformation_artifact=data_transformation_artifact)
        model_pusher_artifact = model_pusher.initated_model_pusher()


    


    except Exception as e:
        raise SensorException(error_message=e,error_detail=sys)
