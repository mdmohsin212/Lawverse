# import numpy as np
# from datasets import Dataset
# from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy
# from ragas import evaluate
# from langchain_core.language_models.base import BaseLanguageModel
# from langchain_core.embeddings import Embeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# class RagasMetrics:
#     def __init__(self):
#         self.metrics = {
#             "context_precision": context_precision,
#             "context_recall": context_recall,
#             "faithfulness": faithfulness,
#             "answer_relevancy": answer_relevancy,
#         }
        
#     def evaluate_dataset(self, dataset : Dataset, llm : BaseLanguageModel, embedding : Embeddings):
#         result = evaluate(
#             dataset=dataset,
#             metrics=list(self.metrics.values()),
#             llm=llm,
#             embeddings=embedding
#         )
#         return {k : round(v, 4) for k, v in result.items()}

#     @staticmethod
#     def f_recall(pred_answer, true_answer):
#         tp = sum(1 for p, t in zip(pred_answer, true_answer) if t.lower() in p.lower())
#         fn = len(true_answer) - tp
#         recall = tp / (tp + fn + 1e-8)
#         return round(recall, 4)

# def compute_all_metrics(dataset : Dataset, preds, trues, llm : BaseLanguageModel):
#     ragas = RagasMetrics()
    
#     hf_embeddings = HuggingFaceEmbeddings(
#         model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
#     )
    
#     ragas_score = ragas.evaluate_dataset(dataset, llm, hf_embeddings)
#     ragas_score["f_recall"] = ragas.f_recall(preds, trues)
#     return ragas_score