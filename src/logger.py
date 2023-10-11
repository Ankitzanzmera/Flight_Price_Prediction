import logging
from datetime import datetime
import os

dirname = f'{datetime.now().strftime("%d_%m_%Y")}'
dirpath = os.path.join(os.getcwd(),'logs',dirname)
os.makedirs(dirpath,exist_ok = True)

log_file_name = f"{datetime.now().strftime('%H_%M_%S')}.log"
log_file_path = os.path.join(dirpath,log_file_name)
print(log_file_name)

logging.basicConfig(
    filename  = log_file_path,
    level = logging.INFO,
    format = "[ %(asctime)s ] %(lineno)d - %(module)s - %(levelname)s - %(message)s " 
)