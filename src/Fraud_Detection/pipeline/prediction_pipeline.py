# prediction pipeline
import sys
import os
from dataclasses import dataclass
from src.Fraud_Detection.utils.common import load_object
from src.Fraud_Detection.logger.logger import logger
from src.Fraud_Detection.exception.exceptions import CustomException

@dataclass
class PredictionPipelineConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def predict(self,features):
        try:
            logger.info(f"Input shape received: {features.shape}")
            model = load_object(file_path=self.prediction_pipeline_config.model_path)
            preprocessor = load_object(file_path=self.prediction_pipeline_config.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            pred_proba = model.predict_proba(data_scaled)[:, 1]
            logger.info(f"Predictions: {preds} and Prediction probabilities: {pred_proba}")
            return preds, pred_proba
        except Exception as e:
            raise CustomException(e,sys)
        
