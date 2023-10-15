import os,sys
sys.path.append(os.getcwd())

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import train_model,hyprtparameter_tuning
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    os.makedirs(os.path.join(os.getcwd(),'artifacts'),exist_ok=True)
    trained_model_path = os.path.join(os.getcwd(),"artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_training(self,train_data,test_data):
        try:
            X_train,y_train = train_data[:,:-1],train_data[:,-1]
            X_test,y_test = test_data[:,:-1],test_data[:,-1]
            logging.info('Divided Independent and Dependent Variable')

            report = train_model(X_train,y_train,X_test,y_test)
            logging.info(f'{report}')
            logging.info(f"Best Model Accuracy Without Tuning is {max(report,key=lambda k :report[k])}")
            model_name,tuned_model,tuned_params = hyprtparameter_tuning(report,X_train,y_train,X_test,y_test)

            save_object(self.model_trainer_config.trained_model_path,tuned_model)

            return model_name,tuned_model,tuned_params,X_test,y_test
        except Exception as e:
            raise CustomException(e,sys)