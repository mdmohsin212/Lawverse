import sys
import os
import json
import time
from datasets import Dataset
from Lawverse.evaluation.metrics import compute_all_metrics
from Lawverse.pipeline.rag_pipeline import rag_components, create_chat_chian
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from langchain_core.language_models.base import BaseLanguageModel
from ragas.llms import LangchainLLMWrapper
from ragas import RunConfig

METRICS_JSON_PATH = "monitoring/rag_metrics.json"

def eval_dataset(eval_data):
    components = rag_components()
    chain, _ = create_chat_chian(components)
    
    eval_results = []
    for sample in eval_data:
        time.sleep(2)
        try:
            result = chain.invoke({"question": sample["question"]})
            answer = result["answer"]
            context_docs = [d.page_content for d in result["source_documents"]]
            
            eval_results.append({
                "question": sample["question"],
                "answer": answer,
                "contexts": context_docs,
                "ground_truth": sample["ground_truth"]
            })
        except Exception as e:
            logging.error(f"Evaluation failed for question '{sample['question']}': {e}")
            raise ExceptionHandle(e, sys)

    return Dataset.from_list(eval_results)

def run_ragas_evaluation(eval_data, llm : BaseLanguageModel):
    dataset = eval_dataset(eval_data)
    preds = [item["answer"] for item in dataset]
    trues = [item["ground_truth"] for item in dataset]
    eval_llm = LangchainLLMWrapper(llm)
    
    run_config = RunConfig(max_workers=4)
    
    results = compute_all_metrics(dataset, preds, trues, eval_llm, run_config)
    
    logging.info(f"RAGAS evaluation completed. Scores: {results}")

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": results
    }
    
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)
    else:
        os.makedirs(os.path.dirname(METRICS_JSON_PATH), exist_ok=True)
        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([entry], f, indent=4)
            
    return results