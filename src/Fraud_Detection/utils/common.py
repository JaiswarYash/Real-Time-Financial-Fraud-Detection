import os
import sys
import numpy as np
import pandas as pd
import dill
from src.Fraud_Detection.logger.logger import logger
from src.Fraud_Detection.exception.exceptions import CustomException
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from mlflo

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        BEST_THRESHOLD = 0.3 # # XGBoost at threshold 0.3: Recall=0.857, Precision=0.894, F1=0.875
        report = {}
        for model_name, model in models.items():
            param = params.get(model_name, {})
            if param:
                logger.info(f"performing hyperparameter tuning for model: {model_name}") # if model_name == "Random_Forest" else 20
                rs = RandomizedSearchCV(estimator=model, param_distributions=param, n_iter=5 if model_name == "Random_Forest" else 10, random_state=42, cv=3, n_jobs=-1)
                rs.fit(X_train, y_train)
                model.set_params(**rs.best_params_)
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)


            y_train_pred = model.predict(X_train)
            y_prob = model.predict_proba(X_test)[:,1]


            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
         
            # threshold = [0.3,0.5,0.6,0.7,0.8,0.9]
            # for t in threshold:
            #     y_test_pred = (y_prob >= t).astype(int)
            #     precision = precision_score(y_test, y_test_pred)
            #     recall = recall_score(y_test, y_test_pred)
            #     f1 = f1_score(y_test, y_test_pred)
            #     logger.info(f"Model: {model_name} - Threshold: {t} - Test Precision: {precision}, Test Recall: {recall}, Test F1: {f1}")
                
            
            y_test_pred_final = (y_prob >= BEST_THRESHOLD).astype(int)
            report[model_name] = {
                "test_precision": precision_score(y_test, y_test_pred_final),
                "test_recall": recall_score(y_test, y_test_pred_final),
                "test_f1": f1_score(y_test, y_test_pred_final),
                "best_threshold": BEST_THRESHOLD
            }
            logger.info(f"Model: {model_name} - Train Precision: {train_precision}, Train Recall: {train_recall}, Train F1: {train_f1}")
            logger.info(f"Model: {model_name} - Test Precision: {report[model_name]['test_precision']}, Test Recall: {report[model_name]['test_recall']}, Test F1: {report[model_name]['test_f1']} at Threshold: {report[model_name]['best_threshold']}")
        return report
    except Exception as e:
            raise CustomException(e, sys)
