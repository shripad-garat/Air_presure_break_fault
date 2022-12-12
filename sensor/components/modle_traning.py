from sensor.entity import config_entity,artifact_entity
from sensor import utility
from sensor.logger import logging
from sensor.exception import SensorException
from sensor import utility
import os,sys
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score



class ModelTraning:

    def __init__(self,model_traning_config:config_entity.ModelTraningConfig,data_transformation_artifact:artifact_entity.DataTransformationArtifact):

        try:
            self.model_traning_config = model_traning_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise SensorException(e,sys)

    def model_selection(self,x,y):
        try:
            logging.info(f"Preparing the model")
            model = XGBClassifier()
            model.fit(x,y)
            return model
            

        except Exception as e:
            raise SensorException(e,sys)



    def inidated_model_traning(self):
        try:
            logging.info(f"Loading the transformed data for tarin and test dataset")
            train_arr = utility.load_numpy_array_data(self.data_transformation_artifact.transformed_train_path)
            test_arr = utility.load_numpy_array_data(self.data_transformation_artifact.transformed_test_path)
            logging.info(f"Spleating the data in to X and Y features for model traning")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            logging.info(f"Selecting the model for predection")
            model = self.model_selection(x=x_train,y=y_train)
            logging.info(f"Getting the F1 score for train dataset")
            y_predect = model.predict(x_train)
            train_score = f1_score(y_true=y_train,y_pred=y_predect)
            logging.info(f"F1 score for train dataset: {train_score}")
            logging.info(f"Getting the F1 score for test dataset")
            y_predect = model.predict(x_test)
            test_score = f1_score(y_true=y_test,y_pred=y_predect)
            logging.info(f"F1 score for test dataset: {test_score}")
            
            logging.info(f"Checking the model F1 score and compariing with expected score")
            if test_score<self.model_traning_config.expected_score:
                raise Exception(f"Model is not good the F1 scored got {test_score} and we expected atleast {self.model_traning_config.expected_score}")
            diffrence = train_score - test_score
            logging.info(f"Checking for Overfitting")
            if diffrence>self.model_traning_config.overfitting_threshold:
                raise Exception("The model is overfitting")
            logging.info(f"Saving the Model")
            utility.save_object(file_path=self.model_traning_config.model_path,obj=model)

            logging.info(f"Preparing the artifact")
            model_traning_artifact = artifact_entity.ModelTraningArtifact(self.model_traning_config.model_path,
                f1_train_score=train_score,
                f1_test_score=test_score
            )
            return model_traning_artifact
               


        except Exception as e:
            raise SensorException(e,sys)