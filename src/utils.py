import os,sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pickle
import pymysql
from tqdm import tqdm
from src.logger import logging 
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_data_from_sql():
    try:
        mydb = pymysql.connect(host=host,user=user,password=password,database=db)
        logging.info('Connection Established')
        df = pd.read_sql_query("select * from flight_price_data",mydb)
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(filepath:str,obj: object):
    try:
        os.makedirs(os.path.join(os.getcwd(),'artifacts'),exist_ok=True)
        with open(filepath,"wb") as  file_obj:
            pickle.dump(obj,file_obj)        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(filepath:str) -> object:
    try:
        with open(filepath,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def train_model(X_train,y_train,X_test,y_test):
    models_list = All_model_list()
    report = {}
    for i in tqdm(range(len(models_list))):
        model = list(models_list.values())[i]

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test,y_pred)

        report[list(models_list.keys())[i]] = r2
    return report

def hyprtparameter_tuning(report: dict,X_train,y_train,X_test,y_test)-> dict:
    model_name = max(report,key=lambda k:report[k])

    model = All_model_list()[model_name]
    params = params_of_models(model_name)

    random_search = RandomizedSearchCV(model,param_distributions=params,cv=5,n_iter=30)
    random_search.fit(X_train,y_train)
    
    tuned_params = random_search.best_params_
    tuned_model = All_model_list()[model_name]
    tuned_model.set_params(**tuned_params)

    tuned_model.fit(X_train,y_train)
    y_pred = tuned_model.predict(X_test)
    r2 = r2_score(y_test,y_pred)

    logging.info(f'Accuracy after tuning is {r2} of model name = {model_name}')
    logging.info('Hyper Parameter has been done...')

    return model_name,tuned_model,tuned_params

def All_model_list():
    models_list = {
        "Linear_Regression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "ElasticNet":ElasticNet(),
        "Decision_Tree":DecisionTreeRegressor(),
        "SVM":SVR(),
        "Random_Forest":RandomForestRegressor(),
        "Gradient_Boosting":GradientBoostingRegressor(),
        "AdaBoost":AdaBoostRegressor(),
        "XGB":XGBRegressor(),
        "Catboost":CatBoostRegressor(),
        "Neighbors":KNeighborsRegressor()
    }
    return models_list

def params_of_models(model_name:str)->dict:
    params = {
        "Linear_Regression": {},
        "Ridge": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        },
        "Lasso": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        },
        "ElasticNet": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9],
        },
        "Decision_Tree": {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        "SVM": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto'] + [0.01, 0.1, 1.0],
        },
        "LinearSVM" :{
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'dual': [True, False]
        },
        "Random_Forest": {
            'n_estimators': [8, 16, 32, 64, 128, 256],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
        },
        "AdaBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
        },
        "Gradient_Boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        "Neighbors": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        },
        "XGB": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
        },
        "Catboost": {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5],
            'border_count': [32, 64, 128],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bylevel': [0.7, 0.8, 0.9],
            'custom_metric': ['RMSE', 'MAE'],
        }
    }
    return params[model_name]