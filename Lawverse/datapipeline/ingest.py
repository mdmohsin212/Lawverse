import sys
import os
import requests
from Lawverse.utils.config import PDF_DIR, PDF_URL
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle

def fetch_file(urls=PDF_URL):
    try:
        paths = []
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        
        for url in urls:
            filename = url.split('/')[-1]
            save_path = PDF_DIR / filename
            
            if not save_path.exists():
                logging.info(f"Downloading {filename} from GitHub Raw...")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                        
                logging.info(f"PDF successfully saved at: {save_path}")
            else:
                logging.info(f"PDF already exists at: {save_path}")

            paths.append(save_path)
        return paths

    except Exception as e:
        logging.error("Data ingestion failed during GitHub PDF fetch!")
        raise ExceptionHandle(e, sys)