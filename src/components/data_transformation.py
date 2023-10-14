import os,sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

@dataclass
class DataTransformationConfig:
    os.makedirs(os.path.join(os.getcwd(),"artifacts"),exist_ok=True)
    peprocessor_path = os.path.join(os.getcwd(),'artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def preprocess_data(self,df):

        ### Preprocess Data According Notebooks

        ## unnamed col
        df.drop(['Unnamed: 0'],axis = 1,inplace = True)

        ## Date of Journey
        df['date'] = df['Date_of_Journey'].str.split("/").str[0].astype(float)
        df['month'] = df['Date_of_Journey'].str.split("/").str[1].astype(float)
        df['year'] = df['Date_of_Journey'].str.split("/").str[2].astype(float)
        df.drop(['Date_of_Journey'],axis = 1,inplace=True)

        ## Route
        df.drop(['Route'],axis = 1,inplace=True)

        ## Departure Time
        df['dep_hour'] = df['Dep_Time'].str.split(":").str[0].astype(float)
        df['dep_min'] = df['Dep_Time'].str.split(":").str[1].astype(float)
        df.drop(['Dep_Time'],axis = 1,inplace=True)

        ## Arrival Time
        df['Arrival_hr'] = df['Arrival_Time'].str.split(":").str[0].astype(float)
        df["Arrival_min"] = df['Arrival_Time'].str.split(":").str[1].str.split(" ").str[0].astype(float)
        df.drop(['Arrival_Time'],axis = 1,inplace=True)

        ## Duration
        df['Duration_hr'] = df['Duration'].str.split(" ").str[0].str.split("h").str[0]
        df['Duration_min'] = df['Duration'].str.split(" ").str[1].str.split("m").str[0].astype(float)
        df['Duration_min'] = df['Duration_min'].fillna(0)

        if "5m" in df['Duration_hr'].values:
            df = df[df['Duration_hr'] != "5m"] 
            df['Duration_hr'] = df['Duration_hr'].astype(float)
        else:
            df['Duration_hr'] = df['Duration_hr'].astype(float)

        df.drop(['Duration'],axis = 1,inplace=True)


        ## Total Stops
        df['Total_Stops'] = df['Total_Stops'].replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4,np.nan:0}).astype(float)

        ## Additional Info
        df.drop(['Additional_Info'],axis = 1,inplace=True)

        return df


    def preprocessing_pipelines(self,cate_features,num_features):
        try:
            categorical_pipeline = Pipeline(
                steps=[
                    ("Impuer",SimpleImputer(strategy="most_frequent")),
                    ("Encoder",OneHotEncoder())
                ]
            )

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            main_pipeline = ColumnTransformer(
                [
                    ("Categorical_pipeline",categorical_pipeline,cate_features),
                    ("Numerical_pipeline",numerical_pipeline,num_features)
                ]
            )

            return main_pipeline
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)           
            logging.info('Got the Training and Test Data')

            cleaned_train_data = self.preprocess_data(train_data)
            cleaned_test_data  = self.preprocess_data(test_data)
            logging.info('Got cleaned Train and Test Data')

            target = "Price"
            cleaned_train_data_input = cleaned_train_data.drop(target,axis = 1)
            cleaned_train_data_target  = cleaned_train_data[target]

            cleaned_test_data_input = cleaned_test_data.drop(target,axis = 1)
            cleaned_test_data_target = cleaned_test_data[target]
            logging.info('Divided the Input variable and Target')


            cate_features = ["Airline","Source","Destination"]
            num_features  = [i for i in cleaned_train_data_input.columns if i not in cate_features]
            
            preprocessing_pipeline = self.preprocessing_pipelines(cate_features,num_features)
            preprocessed_train_data_input = preprocessing_pipeline.fit_transform(cleaned_train_data_input)
            preprocessed_test_data_input = preprocessing_pipeline.transform(cleaned_test_data_input)
            logging.info("Done all Preprocessing")

            train_data = np.c_[preprocessed_train_data_input,cleaned_train_data_target]
            test_data = np.c_[preprocessed_test_data_input,cleaned_test_data_target]
            logging.info("Concatenation of input and target is done")

            save_object(self.data_transformation_config.peprocessor_path,preprocessing_pipeline)

            return (train_data,test_data)

        except Exception as e:
            raise CustomException(e,sys)

