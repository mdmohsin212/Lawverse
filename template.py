import os
from pathlib import Path

project_name = "Lawverse"

list_of_files = [
    f"{project_name}/datapipeline/__init__.py",
    f"{project_name}/datapipeline/ingest.py",
    f"{project_name}/datapipeline/dataset_loader.py",
    f"{project_name}/datapipeline/preprocess.py",
    
    f"{project_name}/retrieval/__init__.py",
    f"{project_name}/retrieval/dense.py",
    f"{project_name}/retrieval/sparse.py",
    f"{project_name}/retrieval/hybrid.py",
    f"{project_name}/retrieval/indexer.py",
    
    f"{project_name}/evaluation/__init__.py",
    f"{project_name}/evaluation/ragas_eval.py",
    f"{project_name}/evaluation/metrics.py",
    
    f"{project_name}/memory/__init__.py",
    f"{project_name}/memory/langchain_memory.py",
    
    f"{project_name}/orchestration/__init__.py",
    f"{project_name}/orchestration/airflow_dag.py",
    
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/config.py",
    f"{project_name}/utils/storage.py",
    
    f"{project_name}/logger/__init__.py",
    f"{project_name}/exception/__init__.py",
    
    f"{project_name}/monitoring/README.md",
    
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
    "config/sources.yaml"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
    else:
        print(f'{filename} is already present in {filedir} and has some content. Skipping creation.')