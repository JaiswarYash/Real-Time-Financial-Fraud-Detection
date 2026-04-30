import os
import sys
from src.Fraud_Detection.pipeline.train_pipeline import TrainPipeline
from src.Fraud_Detection.logger.logger import logger
from src.Fraud_Detection.exception.exceptions import CustomException

def main():
    try:
        logger.info("starting the training pipeline")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logger.info("training pipeline completed")
    except Exception as e:
        logger.error(f"Error occurred in training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()