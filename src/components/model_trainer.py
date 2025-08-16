import os 
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor

)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object, evaluate_models

@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test data ")
            X_train ,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, allow_writing_files=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # parameter grids for hyperparameter tuning (GridSearchCV)
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10, 20]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [None, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [3,5,7]
                },
                "XGBRegressor": {
                    'n_estimators': [50,100],
                    'learning_rate': [0.05, 0.1]
                },
                "CatBoosting Regressor": {
                    'iterations': [100,200],
                    'learning_rate': [0.03, 0.1]
                },
                # leave Linear Regression and AdaBoost with defaults for speed
            }

            model_report, best_models = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get the best model name and estimator
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = best_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name} -> {best_model}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            return score


        except Exception as e:
            raise CustomException(e,sys)