# training-pipeline
import sys
import os
from src.Fraud_Detection.components.data_ingestion import DataIngestion
from src.Fraud_Detection.components.data_transformation import DataTransformation
from src.Fraud_Detection.components.model_training import ModelTrainer
from src.Fraud_Detection.exception.exceptions import CustomException
from src.Fraud_Detection.logger.logger import logger

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info("Training pipeline starts")
            data_ingestion = DataIngestion()
            train_data_path, test_path = data_ingestion.initiate_data_ingestion()
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_path)
            model_trainer = ModelTrainer()
            best_model_name, best_model_score, best_precision = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logger.info(f"Best model: {best_model_name} with recall score: {best_model_score} & precision score: {best_precision}")
            logger.info("Training pipeline completed")
            return best_model_name, best_model_score, best_precision
        except Exception as e:
            raise CustomException(e, sys)