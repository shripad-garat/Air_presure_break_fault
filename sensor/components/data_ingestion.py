from sensor.logger import logging
from sensor.exception import SensorException
import os,sys
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor import utility
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 


class DataInjestion:

    def __init__(self,data_injestion_config:config_entity.DataIngestionConfig):
        try:
            self.data_injestion_config = data_injestion_config
        except Exception as e:
            SensorException(error_message=e,error_detail=sys)

    def initiate_data_injection(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting data as Panda's DataFrame")
            self.df = utility.get_collection_as_dataframe(
                database_name=self.data_injestion_config.Database_name,
                collection_name=self.data_injestion_config.collection_name
            )


            logging.info(f"Replacing 'na' with np.NAN")

            self.df.replace( "na",np.NAN,inplace=True)

            logging.info(f"Creating the traning and testing data with {self.data_injestion_config.test_size*100}% test size")

            self.train,self.test = train_test_split(self.df,test_size= self.data_injestion_config.test_size)

            logging.info(f"Storing the dataset, traning dataset and test dataset in respective dictories")
            feature_store_path =os.path.dirname(self.data_injestion_config.featture_store_file_path)
            os.makedirs(feature_store_path,exist_ok=True)
            self.df.to_csv(path_or_buf=self.data_injestion_config.featture_store_file_path,header=True,index=False)
            data_set_path = os.path.dirname(self.data_injestion_config.test_file_path)
            os.makedirs(data_set_path,exist_ok=True)
            self.train.to_csv(path_or_buf=self.data_injestion_config.train_file_path,header=True,index=False)
            self.test.to_csv(path_or_buf=self.data_injestion_config.test_file_path,header=True,index=False)

            logging.info(f"Creating Data Ingection Artifact ")

            data_injestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_path=self.data_injestion_config.featture_store_file_path,
                test_path=self.data_injestion_config.test_file_path,
                train_path=self.data_injestion_config.train_file_path
            )
            logging.info(f"Data ingestion artifact: {data_injestion_artifact}")
            return data_injestion_artifact




        except Exception as e:
            SensorException(error_message=e,error_detail=sys)

