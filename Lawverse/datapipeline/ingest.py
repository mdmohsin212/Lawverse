import sys
import gdown
from Lawverse.utils.config import PDF_DIR, PDF_URL
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle

def fetch_file(urls=PDF_URL):
    try:
        paths = []
        for url in urls:
            file_id = url.split('/d/')[1].split('/')[0]
            filename = f"{file_id}.pdf"
            save_path = PDF_DIR / filename
            
            if not save_path.exists():
                gdown.download(id=file_id, output=str(save_path), quiet=False)
                logging.info(f"PDF saved at: {save_path}")
            else:
                logging.info(f"PDF already exists at: {save_path}")

            paths.append(save_path)
        return paths

    except Exception as e:
        logging.error("Data ingestion failed during PDF fetch!")
        raise ExceptionHandle(e, sys)