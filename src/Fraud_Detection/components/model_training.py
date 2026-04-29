# model_training
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from src.Fraud_Detection.exception.exceptions import CustomException
from src.Fraud_Detection.logger.logger import logger
from src.Fraud_Detection.utils.common import save_object, evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        
        try:
            logger.info("splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(),
            }

            parameters = {
                 "Logistic_Regression": {
                    'class_weight': 'balanced',
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['libliner', 'saga'],
                    'max_iter': [100, 200, 300]
                 },
                 "Random_forest": {
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_sample_split': [2,5,10],
                    'min_samples_leaf': [1,2,4],
                    'bootstrap': [True, False],
                    'max_features': ['auto', 'sqrt']
                 },
                 "XGBoost": {
                      'n_estimators': [100, 200, 300],
                      'learning_rate': [0.01, 0.1, 0.2, 0.3],
                      'max_depth': [3, 6, 9],
                      'subsample': [0.6,0.8,1.0],
                      'colsample_bytree': [0.6,0.8,1.0],
                      'scale_pos_weight': [100, 200, 500, 1000]
                 }
            }

            model_report: dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=parameters)

            # to get best model name from dict & score
            best_model_name = max(model_report, key=lambda model: model_report[model]['test_recall'])
            best_model = models[best_model_name]

            if model_report[best_model_name]['test_recall'] < 0.3:
                    raise CustomException("No best model found", sys)
                
            logger.info(f"Best found model on both training and testing dataset: {best_model_name} with score: {model_report[best_model_name]['test_recall']}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logger.info("Saved best model object.")

            return best_model_name, model_report[best_model_name]['test_recall']

        except Exception as e:
            raise CustomException(e, sys)