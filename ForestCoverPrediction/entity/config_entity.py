import os,sys
from ForestCoverPrediction.exception import ForestCoverPredictionException
from ForestCoverPrediction.logger import logging
from datetime import datetime

FILE_NAME = "forest_cover.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "Cover_Type"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise ForestCoverPredictionException(e,sys)


class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_file_path = r"notebook/train.csv"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception  as e:
            raise ForestCoverPredictionException(e,sys)

    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise ForestCoverPredictionException(e,sys)

class DataValidationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_validation")
        self.report_file_path=os.path.join(self.data_validation_dir, "report.yaml")
        self.missing_threshold:float = 0.2

class DataTransformationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path =  os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv","npz"))
        self.transformed_test_path =os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv","npz"))

class ModelTrainEvalConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_threshold = 0.1


