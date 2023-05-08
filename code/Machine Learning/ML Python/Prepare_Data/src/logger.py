import os
import logging
import settings as sett

logging_file = os.path.join(sett.dir_log, 'prepare_data.log')
logging.basicConfig(filename=logging_file, level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
