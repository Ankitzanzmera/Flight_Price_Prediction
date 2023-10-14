import os,sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pickle
import pymysql
from src.logger import logging 
from src.exception import CustomException

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
        