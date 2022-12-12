from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity,artifact_entity
from sensor import utility
from sensor.config import TARGET_COLUMN
from typing import Optional
import os,sys
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


class DataValidation:

    def __init__(self,data_validation_config:config_entity.DataValidationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()

        except Exception as e:
            raise SensorException(e,sys)

    def drop_missing_value_column(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        try:
            self.threshold = self.data_validation_config.missing_threshold
            logging.info(f"Droping the missing Value column which have missing values greater than threshold {self.threshold}")
            self.report_missing_value = df.isna().sum()/df.shape[0]
            self.missing_value_column_above_threshold = self.report_missing_value[self.report_missing_value>self.threshold].index
            df.drop(list(self.missing_value_column_above_threshold),axis=1,inplace=True)
            self.validation_error[report_key_name] = self.missing_value_column_above_threshold
            logging.info(f"Droping the columns: {self.missing_value_column_above_threshold}")
            if len(df.columns)==0:
                return None
            return df

        except Exception as e:
            raise SensorException(e,sys)

    def is_required_column_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            base_df_column = base_df.columns
            current_df_column = current_df.columns
            missing_column = list()
            logging.info(f"Checking the required columns are present in the current dataset")
            for column in current_df_column:
                if column not in base_df_column:
                    missing_column.append(column)

            if len(missing_column)>0:
                self.validation_error[report_key_name] = False
                self.validation_error[report_key_name+"missing_columns"] = missing_column
                logging.info(f"Missing columns are : {missing_column}")
                return False
            self.validation_error[report_key_name] = True
            return True

        except Exception as e:
            raise SensorException(e,sys)


    def data_drift(self,base_df:pd.DataFrame,curent_df:pd.DataFrame,report_key_name:str):
        try:
            base_columns = base_df.columns
            data_drift_report = dict()
            logging.info(f"Checking for if there is any data drift in current dataset")
            for column in base_columns:
                curent_column_df,base_column_df = curent_df[column],base_df[column]
                same_dstribution = ks_2samp(curent_column_df,base_column_df)

                if same_dstribution.pvalue>0.05:
                    data_drift_report[column]={
                        "P_value":same_dstribution.pvalue,
                        "same_dstribution":True
                    }
                else:
                    data_drift_report[column]={
                        "p_value":same_dstribution.pvalue,
                        "same_dstripution":False
                    }
                    logging.info(f"The column: {column} in current dataframe is having data drift with p_value: {same_dstribution.pvalue}")
                

            self.validation_error[report_key_name] = data_drift_report

        except Exception as e:
            raise SensorException(e,sys)



    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Initiating the data validation")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            logging.info(f"reading the base dataset")
            base_df.replace('na',np.NAN,inplace=True)
            train_df = pd.read_csv(self.data_ingestion_artifact.train_path)
            logging.info(f"reading the train dataset")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_path)
            logging.info(f"reading the test dataset")
            base_df = self.drop_missing_value_column(base_df,"missing_value_column_in_base_df")
            train_df = self.drop_missing_value_column(train_df,"missing_value_column_in_train_df")
            test_df = self.drop_missing_value_column(test_df,"missing_value_column_in_test_df")
            logging.info(f"Converting the datatype of dataset into float datatype")
            exclude_columns = [TARGET_COLUMN]
            base_df = utility.convert_to_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utility.convert_to_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utility.convert_to_float(df=test_df, exclude_columns=exclude_columns)
            logging.info(f"Checking the status of validation on the datasets")
            if self.is_required_column_exists(base_df,train_df,"missing_column_in_train_df"):
                self.data_drift(base_df,train_df,"drift_report_train_df")
            if self.is_required_column_exists(base_df,test_df,"missing_cloumn_in_test_df"):
                self.data_drift(base_df,test_df,"drift_report_test_data")

            logging.info("Writing the report into yaml file for further refrence")
            utility.write_yaml_report(file_path=self.data_validation_config.report_file_path,data=self.validation_error)
            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"setting the data validation artifact")
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e,sys)