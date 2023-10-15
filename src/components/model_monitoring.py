import os,sys
sys.path.append(os.getcwd())
import numpy as np
from src.exception import CustomException
from src.logger import logging
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


class ModelMonitoring:
    def eval_metrics(self,y_test,y_pred):
        mse = mean_squared_error(y_test,y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)
        return {"MSE":mse,"MAE":mae,"RMSE":rmse,"r2_score":r2}
    
    def initiate_model_monotoring(self,model_name,tuned_model,best_params,X_test,y_test):
        try:
            mlflow.set_registry_uri("https://dagshub.com/Ankitzanzmera/flight_price_prediction.mlflow")
            tracking_uri_type_score = urlparse(mlflow.get_registry_uri()).scheme

            with mlflow.start_run():
                y_pred = tuned_model.predict(X_test)

                metrics = self.eval_metrics(y_test,y_pred)
                for i,j in zip(list(metrics.keys()),list(metrics.values())):
                    mlflow.log_metric(i,j)
                logging.info('Metrics uploaded')

                for i,j in zip(list(best_params.keys()),list(best_params.values())):
                    mlflow.log_param(i,j)
                logging.info("Params uploaded")

                if tracking_uri_type_score != 'file':
                    mlflow.sklearn.log_model(tuned_model,"Model",registered_model_name=model_name)
                    logging.info('Model uploaded')
                else:
                    logging.info("Something Unexpected occurred..")
        except Exception as e:
            raise CustomException(e,sys)



