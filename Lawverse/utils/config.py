import os
from pathlib import Path

# ---- Local ----
# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "data"
# PDF_DIR = DATA_DIR / "raw"
# PROCESSED_DIR = DATA_DIR / "process"
# MEMORY_DIR = DATA_DIR / "memory/store"
# FAISS_PATH = DATA_DIR / "faiss_index"


PDF_URL = [
    "https://raw.githubusercontent.com/mdmohsin212/Kaggle-competitions/main/Dataset/Digital-Security-Act-2018.pdf",
    "http://raw.githubusercontent.com/mdmohsin212/Kaggle-competitions/main/Dataset/Bangladesh%20COMPANIES%20ACT.pdf",
    "http://raw.githubusercontent.com/mdmohsin212/Kaggle-competitions/main/Dataset/Bangladesh-Labour-Act-2018.pdf"
]

# ---- HF ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/tmp/lawverse_data")

PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "process"
MEMORY_DIR = DATA_DIR / "memory/store"

FAISS_PATH = DATA_DIR / "faiss_index"

for data in [DATA_DIR, PDF_DIR, PROCESSED_DIR, MEMORY_DIR]:
    os.makedirs(data, exist_ok=True)