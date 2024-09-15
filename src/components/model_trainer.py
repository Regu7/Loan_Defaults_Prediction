import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                # "Logistic Regression": LogisticRegression(),
                # "Random Forest": RandomForestClassifier(),
                "xgboost": XGBClassifier(),
            }

            params = {
                # "Logistic Regression": {
                #     "penalty": ["l1", "l2", "elasticnet", "none"],
                #     "C": [0.01, 0.1, 1, 10, 100],
                #     "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                #     "max_iter": [100, 200, 300],
                # },
                # "Random Forest": {
                #     "n_estimators": [100, 200, 300],
                #     "criterion": ["gini", "entropy"],
                #     "max_depth": [None, 10, 20, 30],
                #     "min_samples_split": [2, 5, 10],
                #     "min_samples_leaf": [1, 2, 4],
                #     "bootstrap": [True, False],
                # },
                # "xgboost": {
                #     "n_estimators": [100, 200, 300],
                #     "learning_rate": [0.01, 0.1, 0.2],
                #     "max_depth": [3, 6, 9],
                #     "subsample": [0.6, 0.8, 1.0],
                #     "colsample_bytree": [0.6, 0.8, 1.0],
                #     "gamma": [0, 0.1, 0.2],
                # },
                "xgboost": {
                   # "n_estimators": [100],
                    # "learning_rate": [0.01],
                    # "max_depth": [3, 6, 9],
                    # "subsample": [0.6, 0.8, 1.0],
                    # "colsample_bytree": [0.6, 0.8, 1.0],
                    # "gamma": [0, 0.1, 0.2],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            logging.info("Model Training Completed")

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # if best_model_score < 0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            roc_auc_score_ = roc_auc_score(y_test, predicted)
            return roc_auc_score_

        except Exception as e:
            raise CustomException(e, sys)
