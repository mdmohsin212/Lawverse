import numpy as np
from datasets import Dataset
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy
from ragas import evaluate, RunConfig
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class RagasMetrics:
    def __init__(self):
        self.metrics = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }
        
    def evaluate_dataset(self, dataset : Dataset, llm : BaseRagasLLM, embedding : BaseRagasEmbeddings, run_config : RunConfig):
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

    @staticmethod
    def f_recall(pred_answer, true_answer):
        pred_tokens = set(" ".join(pred_answer).lower().split())
        true_tokens = set(" ".join(true_answer).lower().split())

        tp = len(pred_tokens & true_tokens)
        fn = len(true_tokens - pred_tokens)

        return round(tp / (tp + fn + 1e-8), 4)


def compute_all_metrics(dataset : Dataset, preds, trues, llm : BaseRagasLLM, run_config : RunConfig):
    ragas = RagasMetrics()
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    
    ragas_score = ragas.evaluate_dataset(dataset, llm, hf_embeddings, run_config)
    ragas_score["f_recall"] = RagasMetrics.f_recall(preds, trues)
    return ragas_score