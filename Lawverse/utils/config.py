import sys
import yaml
import os
from pathlib import Path
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle

def load_config(path : str):
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Config loaded from path {path}")
        return config

    except Exception as e:
        logging.erorr(f"Error loading config {path}")
        raise ExceptionHandle(e, sys)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for data in [DATA_DIR, PDF_DIR, PROCESSED_DIR]:
    os.makedirs(data, exist_ok=True)
    
PDF_URL = "https://www.icsi.edu/media/portals/86/bare%20acts/Bangladesh%20COMPANIES%20ACT.pdf"