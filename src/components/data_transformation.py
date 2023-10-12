import os,sys
sys.path.append(os.getcwd())

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    os.makedirs(os.path.join(os.getcwd(),"artifacts"),exist_ok=True)
    peprocessor_path = os.path.join(os.getcwd(),'artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def make_pipelines(self,cate_features,num_features):
        pass

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)

