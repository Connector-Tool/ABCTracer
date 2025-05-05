import os

CUR_FILE_PATH = __file__
CUR_DIR_PATH = os.path.dirname(CUR_FILE_PATH)
RAW_DIR_PATH = os.path.join(CUR_DIR_PATH, 'raw')
os.makedirs(RAW_DIR_PATH, exist_ok=True)
