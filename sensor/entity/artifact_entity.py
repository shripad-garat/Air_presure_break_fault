from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path:str 
    train_path:str
    test_path:str


@dataclass
class DataValidationArtifact:
    report_file_path:str



@dataclass
class DataTransformationArtifact:
    transformation_object_path:str
    transformed_train_path:str
    transformed_test_path:str
    target_encoder_path:str

@dataclass
class ModelTraningArtifact:
    train_model_path:str
    f1_train_score:str
    f1_test_score:str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    imprued_accuarcy:float

@dataclass
class ModelPusherArtifact:
    pusher_dir:str
    saved_model_dir:str