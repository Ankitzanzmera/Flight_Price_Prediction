import os,sys
sys.path.append(os.getcwd())

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    try:
        logging.info('Execution Has Started')

        data_ingestion = DataIngestion()
        (train_data_path,test_data_path) = data_ingestion.initiate_data_ingestion()
        logging.info('Data Ingestion Has Completed')
        logging.info("-"*35)

        data_transformation = DataTransformation()
        train_data,test_data = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        logging.info('Data Transformation Has Completed')
        logging.info("-"*35)
    except Exception as e:
        raise CustomException(e,sys)