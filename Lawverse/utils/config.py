import os
from pathlib import Path


# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "data"
# PDF_DIR = DATA_DIR / "raw"
# PROCESSED_DIR = DATA_DIR / "process"
# MEMORY_DIR = DATA_DIR / "memory/store"
# FAISS_PATH = DATA_DIR / "faiss_index"

# ---- For Hugging spaces ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/tmp/lawverse_data")

PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "process"
MEMORY_DIR = DATA_DIR / "memory/store"

FAISS_PATH = DATA_DIR / "faiss_index"

for data in [DATA_DIR, PDF_DIR, PROCESSED_DIR, MEMORY_DIR]:
    os.makedirs(data, exist_ok=True)
    
PDF_URL = [
    "https://drive.google.com/file/d/1yVcd9xJPBi03QP0HlGKN56DfJeUlk0fo/view?usp=drive_link",
    "https://drive.google.com/file/d/1OyReUjwjZfDWNGPgP75qSm71aDAwU4ei/view?usp=drive_link",
    "https://drive.google.com/file/d/18EpzwVwGDEXfUmEXhwIXjbQ8KTJZSaye/view?usp=drive_link"
]