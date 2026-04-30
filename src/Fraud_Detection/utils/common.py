import os
import sys
import numpy as np
import pandas as pd
import dill
from src.Fraud_Detection.logger.logger import logger
from src.Fraud_Detection.exception.exceptions import CustomException
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

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
        report = {}
        for model_name, model in models.items():
            param = params.get(model_name, {})
            if param:
                logger.info("performing hyperparameter tuning for model: {model_name}")
                rs = RandomizedSearchCV(estimator=model,param_distributions=param, n_iter=10,random_state=42, cv=5, n_jobs=-1)
                rs.fit(X_train, y_train)
                model.set_params(**rs.best_params_)
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)


            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_roc_auc = roc_auc_score(y_train, y_train_pred)

            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_pred)

            report[model_name] = {
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "train_roc_auc": train_roc_auc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_roc_auc": test_roc_auc
            }
            logger.info(f"Model: {model_name} - Train Precision: {train_precision}, Train Recall: {train_recall}, Train F1: {train_f1}, Train ROC AUC: {train_roc_auc}")
            logger.info(f"Model: {model_name} - Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1: {test_f1}, Test ROC AUC: {test_roc_auc}")
        return report
    except Exception as e:
        raise CustomException(e, sys)
