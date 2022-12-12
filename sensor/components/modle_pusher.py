from sensor.entity import artifact_entity,config_entity
from sensor.logger import logging
from sensor.exception import SensorException
from sensor import utility
from sensor.predector import ModelResolver
import os,sys



class ModelPusher:

    def __init__(self,model_pusher_config:config_entity.ModelPusherConfig,model_traning_artifact:artifact_entity.ModelTraningArtifact,data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_traning_artifact = model_traning_artifact
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise SensorException(e,sys)


    def initated_model_pusher(self):
        try:
            logging.info("Loading the transformer object, target encoder object and model object")
            transformer = utility.load_object(self.data_transformation_artifact.transformation_object_path)
            target_encoder = utility.load_object(self.data_transformation_artifact.target_encoder_path)
            model = utility.load_object(self.model_traning_artifact.train_model_path)

            logging.info("Saving the objects in to pusher directory")
            utility.save_object(self.model_pusher_config.pusher_model_path,obj=model)
            utility.save_object(self.model_pusher_config.pusher_transformer_path,obj=transformer)
            utility.save_object(self.model_pusher_config.pusher_target_encode_path,obj=target_encoder)
            logging.info("Getting the path for saving the Object")
            transformer_path = ModelResolver().get_latest_saved_transformer_path()
            target_encoder_path = ModelResolver().get_latest_saved_target_encoder_path()
            model_path = ModelResolver().get_latest_saved_model_path()
            logging.info('saving the object in to saved_models directory')
            utility.save_object(model_path,obj=model)
            utility.save_object(transformer_path,obj=transformer)
            utility.save_object(target_encoder_path,obj=target_encoder)
            logging.info('Preparing the artifact')
            model_pusher_artifact = artifact_entity.ModelPusherArtifact(
                pusher_dir=self.model_pusher_config.model_pusher_dir,
                saved_model_dir= ModelResolver().get_latest_dir_path
            )
            return model_pusher_artifact


        except Exception as e:
            raise SensorException(e,sys)