import os
from pathlib import Path


# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "data"
# PDF_DIR = DATA_DIR / "raw"
# PROCESSED_DIR = DATA_DIR / "processed"

# ---- For Hugging spaces ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/tmp/lawverse_data")
PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MEMORY_DIR = DATA_DIR / "memory/store"
FAISS_PATH = "/tmp/faiss_index"

for data in [DATA_DIR, PDF_DIR, PROCESSED_DIR, MEMORY_DIR]:
    os.makedirs(data, exist_ok=True)
    
PDF_URL = "https://www.icsi.edu/media/portals/86/bare%20acts/Bangladesh%20COMPANIES%20ACT.pdf"