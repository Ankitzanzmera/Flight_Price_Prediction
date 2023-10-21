import os,sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        self.preprocessor_path = os.path.join(os.getcwd(),"artifacts","preprocessor.pkl")
        self.model_path = os.path.join(os.getcwd(),"artifacts","model.pkl")
    
    def predict(self,input_data):
        try:
            preprocessor_obj = load_object(self.preprocessor_path)
            model_obj = load_object(self.model_path)
            logging.info("Successfullt loaded Preorcessor and Model Object")

            scaled_data = preprocessor_obj.transform(input_data)

            pred = model_obj.predict(scaled_data)
            return pred

        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,Airline,Source,Destination,Date_of_Journey,Total_Stops,Departure_Time) -> None:
        self.Airline = Airline
        self.Source = Source
        self.Destination = Destination
        self.Date_of_Journey = Date_of_Journey
        self.Total_Stops = Total_Stops
        self.Departure_Time = Departure_Time
        logging.info(f'{self.Airline},{self.Source},{self.Destination},{self.Date_of_Journey},{self.Total_Stops},{self.Departure_Time}')

    def preprocess_predict_data(self):
        try:
            self.year = self.Date_of_Journey.split("-")[0]
            self.month = self.Date_of_Journey.split("-")[1]
            self.date = self.Date_of_Journey.split("-")[2]
            del self.Date_of_Journey
            self.dep_hour = self.Departure_Time.split(":")[0]
            self.dep_min = self.Departure_Time.split(":")[1]
            del self.Departure_Time
            # self.Duration_hr = self.Duration.split(":")[0]
            # self.Duration_min = self.Duration.split(":")[0]
            # logging.info(f'{self.Airline},{self.Source},{self.Destination},{self.year,self.month,self.date},{self.Total_Stops},{self.dep_hour,self.dep_min},{self.Duration_hr,self.Duration_min}')
            logging.info("Preprocess of Input Data is Done")
        except Exception as e:
            raise CustomException(e,sys)

    def get_data_as_dataframe(self):
        try:
            input_df = {
                "Airline":[self.Airline],
                "Source":[self.Source],
                "Destination":[self.Destination],
                "year":[self.year],
                "month":[self.month],
                "date":[self.date],
                "Total_Stops":[self.Total_Stops],
                "dep_hour":[self.dep_hour],
                "dep_min":[self.dep_min],
            }
            input_df = pd.DataFrame(input_df)           
            logging.info(f'{input_df}')
            logging.info("Made A Dataframe of Input Data")
            return input_df
        except Exception as e:
            raise CustomException(e,sys)
