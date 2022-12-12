from sensor.logger import logging
from sensor.exception import SensorException
from sensor.predector import ModelResolver
from sensor.entity import config_entity,artifact_entity
from sensor import utility
from sklearn.metrics import f1_score
from sensor.config import TARGET_COLUMN
import pandas as pd 
import os,sys



class ModelEvaluation:

    def __init__(self,model_evaluation_config:config_entity.ModelEvaluationConfig,model_traning_artifact:artifact_entity.ModelTraningArtifact,data_transformation_artifact:artifact_entity.DataTransformationArtifact,data_ingestion_artifat:artifact_entity.DataIngestionArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.model_traning_artifact = model_traning_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifat
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise SensorException(e,sys)


    def initated_model_evaluation(self):
        try:
            logging.info("Getting the latest directory path")
            latest_dir = self.model_resolver.get_latest_dir_path()
            logging.info("checking If the latest directory is empty or not")
            if latest_dir is None:
                logging.info("returning the artifact as directory is empty")
                return artifact_entity.ModelEvaluationArtifact(True,None)
            logging.info("Importing the Transformer object, target_encoder object and model object from saved model")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()
            model_path = self.model_resolver.get_latest_model_path()

            transformer = utility.load_object(transformer_path)
            target_encoder = utility.load_object(target_encoder_path)
            model = utility.load_object(model_path)
            logging.info("Importing the current transformer object, target encoder object and current model")
            current_transformer = utility.load_object(self.data_transformation_artifact.transformation_object_path)
            current_target_encoder = utility.load_object(self.data_transformation_artifact.target_encoder_path)
            current_model = utility.load_object(self.model_traning_artifact.train_model_path)
            logging.info("Importing the test dataset to test the current model and saved model")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_path)
            logging.info("Preparing the data using transformer object and target encoder object for the testing saved model")
            y_true = target_encoder.transform(test_df[TARGET_COLUMN])
            x_test = transformer.transform(test_df.drop(TARGET_COLUMN,axis=1))
            y_predict = model.predict(x_test)
            logging.info("Testing the saved model and geting the score of it")
            score = f1_score(y_true=y_true,y_pred=y_predict)
            logging.info(f"Saved model score: {score}")
            y_true_current = current_target_encoder.transform(test_df[TARGET_COLUMN])
            x_test_current = transformer.transform(test_df.drop(TARGET_COLUMN,axis=1))
            y_predict_current = model.predict(x_test_current)
            logging.info("Testing the current model and geting the score of it")
            current_score = f1_score(y_true=y_true_current,y_pred=y_predict_current)
            logging.info(f"Current model score: {current_score}")

            logging.info("Checking if the current model is performing well than previous model")
            if current_score<score:
                raise Exception(f"The current model is not performing well than previous model")
            logging.info("Preparing the artifact")
            modle_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                imprued_accuarcy=current_score
            )
            return modle_evaluation_artifact

            
        except Exception as e:
            raise SensorException(e,sys)