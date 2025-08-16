import os 
import sys 
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd 
from src.exception import CustomException
import dill
def save_object(file_path,obj):
    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None, cv=3, scoring='r2'):
    """
    Train and evaluate models. If a params dict is provided and contains a key matching a model name,
    a GridSearchCV will be run to tune that model.

    Returns:
      report: dict mapping model name -> test score
      best_models: dict mapping model name -> fitted estimator (best found or original)
    """
    try:
        report = {}
        best_models = {}

        for name, model in models.items():
            if params and name in params and params[name]:
                # run grid search for this model
                gs = GridSearchCV(estimator=model, param_grid=params[name], cv=cv, scoring=scoring, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_estimator = gs.best_estimator_
                fitted_model = best_estimator
            else:
                # fit default model
                fitted_model = model
                fitted_model.fit(X_train, y_train)

            # predictions and scoring
            y_test_pred = fitted_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
            best_models[name] = fitted_model

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        

    except Exception as e:
        raise CustomException(e,sys)