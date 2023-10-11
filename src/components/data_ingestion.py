import os,sys
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException
from src.utils import read_data_from_sql
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    os.makedirs(os.path.join(os.getcwd(),"artifacts"),exist_ok=True)
    raw_data_path = os.path.join(os.getcwd(),'artifacts',"raw.csv")
    train_data_path = os.path.join(os.getcwd(),"artifacts","train.csv")
    test_data_path = os.path.join(os.getcwd(),"artifacts","test.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = read_data_from_sql()
            logging.info("Got the Data from Sql Workbench")

            df.to_csv(self.data_ingestion_config.raw_data_path)

            train_data,test_data = train_test_split(df,test_size=0.2,random_state=45,shuffle=True)

            train_data.to_csv(self.data_ingestion_config.train_data_path)
            test_data.to_csv(self.data_ingestion_config.test_data_path)
            return (self.data_ingestion_config.train_data_path,self.data_ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)

