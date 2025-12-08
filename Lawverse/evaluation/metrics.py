import numpy as np
from datasets import Dataset
from ragas.metrics import context_recall, answer_relevancy, faithfulness
from ragas import evaluate, RunConfig
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def mrr_score(preds, trues):
    ranks = []
    for pred, true in zip(preds, trues):
        rank = 0
        for i, p in enumerate(pred, start=1):
            if p == true:
                rank = i
                break
        ranks.append(1 / rank if rank > 0 else 0)
    return round(float(np.mean(ranks)), 4)

class RagasMetrics:
    def __init__(self):
        self.metrics = {
            "recall@10": context_recall,
            "ndcg@10": answer_relevancy,
            "faithfulness": faithfulness,
        }
        
    def evaluate_dataset(self, dataset: Dataset, llm: BaseRagasLLM, embedding: BaseRagasEmbeddings, run_config: RunConfig):
        result = evaluate(
            dataset=dataset,
            metrics=list(self.metrics.values()),
            llm=llm,
            embeddings=embedding,
            run_config=run_config
        )
        
        if isinstance(result.scores, dict):
            scores_dict = result.scores
        elif isinstance(result.scores, list):
            scores_dict = {k: v for d in result.scores for k, v in d.items()}
        else:
            raise TypeError(f"Unexpected type for result.scores: {type(result.scores)}")

        return {k: round(v, 4) for k, v in scores_dict.items()}

def compute_all_metrics(dataset: Dataset, preds, trues, llm: BaseRagasLLM, run_config: RunConfig):
    ragas = RagasMetrics()
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    
    ragas_score = ragas.evaluate_dataset(dataset, llm, hf_embeddings, run_config)
    ragas_score["mrr"] = mrr_score(preds, trues)
    return ragas_score