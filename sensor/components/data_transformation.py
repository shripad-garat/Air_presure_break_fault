from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity,artifact_entity
from sensor.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sensor import utility
import pandas as pd
import numpy as np
import os,sys





class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise SensorException(e,sys)

    @classmethod
    def get_data_transformation_pipeline(cls)->Pipeline:
        try:
            logging.info(f"Preparing the Simple Imputer to replace Null value")
            simple_imputer = SimpleImputer(strategy="constant",fill_value=0)
            logging.info(f"Preparing Robust Scaller to scaled the data")
            robust_scaler = RobustScaler()
            logging.info(f"Preparing the transformation pipeline for the transformation")
            transformation_pipeline = Pipeline(
                steps={
                    ("IMPUTER",simple_imputer),
                    ("SCALER",robust_scaler)

            })
            return transformation_pipeline

        except Exception as e:
            raise SensorException(e,sys)


    

    def initiate_data_transformation(self):
        try:
            logging.info(f"Importing the tarin and test data")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_path)

            logging.info(f"Seprating the indepandent features and target data")
            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)

            train_target = train_df[TARGET_COLUMN]
            test_target = test_df[TARGET_COLUMN]
            
            logging.info("Encoding the target label into the numerical data")
            encoder = LabelEncoder()
            encoder.fit(train_target)
            encoded_train_target = encoder.transform(train_target)
            encoded_test_target = encoder.transform(test_target)

            logging.info(f"Inicating the transformation pipeline")
            transformation_pipeline = DataTransformation.get_data_transformation_pipeline()
            transformation_pipeline.fit(input_feature_train_df)

            logging.info(f"Transforming Input tarin data")
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            logging.info(f"Transforming Input test data")
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)
            logging.info(f"Balancing the Data")
            smt = SMOTETomek(sampling_strategy = "minority")
            logging.info(f"Befor resampling Input : {input_feature_train_arr.shape}, Target : {encoded_train_target.shape}")
            input_feature_train_arr, encoded_train_target_arr = smt.fit_resample(input_feature_train_arr,encoded_train_target)
            logging.info(f"After resampling Input : {input_feature_train_arr.shape}, Target : {encoded_train_target_arr.shape}")
            logging.info(f"Befor resampling Input : {input_feature_test_arr.shape}, Target : {encoded_test_target.shape}")
            input_feature_test_arr, encoded_test_target_arr = smt.fit_resample(input_feature_test_arr,encoded_test_target)
            logging.info(f"After resampling Input : {input_feature_test_arr.shape}, Target : {encoded_test_target_arr.shape}")
            logging.info("Saving the transformed data")
            train_arr = np.c_[input_feature_train_arr,encoded_train_target_arr]
            test_arr = np.c_[input_feature_test_arr,encoded_test_target_arr]

            utility.save_numpy_array_data(self.data_transformation_config.transformed_train_path,train_arr)
            utility.save_numpy_array_data(self.data_transformation_config.transformed_test_path,test_arr)
            logging.info(f"Saving the transformation pipeline and target encoder")
            utility.save_object(self.data_transformation_config.transformer_object_path,obj= transformation_pipeline)
            utility.save_object(self.data_transformation_config.target_encoder_path,obj= encoder )

            logging.info(f"Preparing the artifact")
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformation_object_path=self.data_transformation_config.transformer_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path
            )

            return data_transformation_artifact



        except Exception as e:
            raise SensorException(e,sys)