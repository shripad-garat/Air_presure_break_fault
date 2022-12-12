from sensor.entity import artifact_entity,config_entity
from sensor.logger import logging
from sensor.exception import SensorException
import os,sys
from typing import Optional
from glob import glob


class ModelResolver:

    def __init__(self,model_registry="svaed_models",
    transformer_dir_name="trransformer",
    model_dir_name="model",
    target_encoder_dir_name="target_encoder"):
        try:
            self.model_registry = model_registry
            self.transformer_dir_name = transformer_dir_name
            self.model_dir_name = model_dir_name
            self.target_encoder_dir_name = target_encoder_dir_name
            os.makedirs(self.model_registry,exist_ok=True)

        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_name = os.listdir(self.model_registry)
            if len(dir_name)==0:
                return None
            dir_name = list(map(int,dir_name))
            latest_dir_name = max(dir_name)
            return os.path.join(self.model_registry,f"{latest_dir_name}")

        except Exception as e:
            raise SensorException(e,sys)      

    def get_latest_model_path(self):
        try:
            latest_dir_path = self.get_latest_dir_path()
            if latest_dir_path is None:
                raise Exception(f"The model is not avalable in Directory")
            return os.path.join(latest_dir_path,self.model_dir_name,config_entity.MODEL_FILE_NAME)
        except Exception as  e:
            raise SensorException(e,sys)  
    def get_latest_transformer_path(self):
        try:
            latest_dir_name = self.get_latest_dir_path()
            if latest_dir_name is None:
                raise Exception(f"The transformer is not avalable in Directory")
            return os.path.join(latest_dir_name,self.transformer_dir_name,config_entity.TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e,sys)
    
    def get_latest_target_encoder_path(self):
        try:
            latest_dir_name = self.get_latest_dir_path()
            if latest_dir_name is None:
                raise Exception(f"The target encoder is not in directory")
            return os.path.join(latest_dir_name,self.target_encoder_dir_name,config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e,sys)


    def get_latest_svaed_dir_path(self)->Optional[str]:
        try:
            latest_dir_name = self.get_latest_dir_path()
            if latest_dir_name is None:
                return os.path.join(self.model_registry,f"{0}")
            latest_dir_num = int(os.path.basename(latest_dir_name))
            return os.path.join(self.model_registry,f"{latest_dir_num+1}")

        except Exception as e:
            raise Exception(e,sys)


    def get_latest_saved_model_path(self):
        try:
            latest_svaed_dir = self.get_latest_svaed_dir_path()
            return os.path.join(latest_svaed_dir,self.model_dir_name,config_entity.MODEL_FILE_NAME)

        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_saved_transformer_path(self):
        try:
            latest_saved_dir = self.get_latest_svaed_dir_path()
            return os.path.join(latest_saved_dir,self.transformer_dir_name,config_entity.TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e,sys)

    def get_latest_saved_target_encoder_path(self):
        try:
            latest_saved_dir = self.get_latest_svaed_dir_path()
            return os.path.join(latest_saved_dir,self.target_encoder_dir_name,config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e,sys)