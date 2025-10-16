import os
import requests
import sys
from Lawverse.utils.config import PDF_DIR, PDF_URL
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle

def fetch_file(url=PDF_URL):
    try:
        filename = os.path.basename(url.split("#")[0])
        save_path = PDF_DIR / filename
        
        if not save_path.exists():
            data = requests.get(url)
            with open(save_path, "wb") as f:
                f.write(data.content)
            logging.info(f"PDF saved at: {save_path}")
        else:
            logging.info(f"PDF already exists at: {save_path}")
        return str(save_path)
    
    except Exception as e:
        logging.error("Data Ingest Failed!!")
        raise ExceptionHandle(e, sys)