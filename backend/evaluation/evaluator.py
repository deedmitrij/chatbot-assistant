import json
import os
import pandas as pd
from typing import List, Dict
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AsyncOpenAI
from datasets import Dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.managers.chat_manager import ChatManager
from backend.managers.factory import create_app_manager
from config import HF_API_TOKEN, HF_BASE_URL, CHAT_MODEL, EMBEDDING_MODEL


class RAGEvaluator:
    """
    A class to evaluate the RAG pipeline performance using the RAGAS framework.
    It runs the evaluation dataset through the ChatManager and computes
    metrics for faithfulness, relevance, and context quality.
    """

    def __init__(self, chat_manager: ChatManager):
        """
        Initializes the evaluator with a ChatManager instance.

        Args:
            chat_manager (ChatManager): The active instance of the hotel chat manager.
        """
        self.chat_manager = chat_manager
        # Define metrics to be used by RAGAS
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        self.llm = self._build_ragas_llm()
        self.embeddings = self._build_ragas_embeddings()

    def _build_ragas_llm(self):
        client = AsyncOpenAI(api_key=HF_API_TOKEN, base_url=HF_BASE_URL)
        return llm_factory(model=CHAT_MODEL, client=client)

    def _build_ragas_embeddings(self):
        hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return LangchainEmbeddingsWrapper(hf_embeddings)

    def load_evaluation_data(self, file_path: str) -> List[Dict[str, str]]:
        """
        Loads the 'golden' dataset from a JSON file.

        Args:
            file_path (str): Path to the eval_dataset.json.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing questions and ground truths.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def prepare_ragas_dataset(self, test_data: List[Dict[str, str]]) -> Dataset:
        """
        Executes questions through the RAG pipeline and prepares the RAGAS Dataset.

        Args:
            test_data (List[Dict[str, str]]): The loaded ground truth data.

        Returns:
            Dataset: A HuggingFace Dataset object formatted for RAGAS evaluation.
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        print(f"🚀 Starting evaluation for {len(test_data)} queries...")

        for entry in test_data:
            query = entry["question"]
            gt = entry["ground_truth"]

            eval_result = self.chat_manager.process_message_for_eval(query)

            questions.append(query)
            answers.append(eval_result["answer"])
            contexts.append([eval_result["context"]]) # RAGAS expects contexts as a list of strings for each question
            ground_truths.append(gt)

            print(f"✅ Processed: {query[:30]}...")

        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        return Dataset.from_dict(data_dict)

    def run(self, dataset_path: str, output_path: str = "results/evaluation_results.csv") -> pd.DataFrame:
        """
        Main execution method to load data, run evaluation, and save results.

        Args:
            dataset_path (str): Path to the input JSON dataset.
            output_path (str): Path to save the resulting scores.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation scores per question.
        """
        test_data = self.load_evaluation_data(dataset_path)
        ragas_dataset = self.prepare_ragas_dataset(test_data)

        # Run RAGAS evaluation
        result = evaluate(
            dataset=ragas_dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )

        df = result.to_pandas()
        df.to_csv(output_path, index=False)

        print(f"\n✨ Evaluation Complete! Results saved to {output_path}")
        print("\n--- Summary Metrics ---")
        print(df.mean(numeric_only=True))

        return df


if __name__ == "__main__":
    chat_manager = create_app_manager(load_data=False)
    evaluator = RAGEvaluator(chat_manager)
    evaluator.run("eval_dataset.json", output_path="results/evaluation_results.csv")
    evaluator.run("eval_negative.json", output_path="results/evaluation_safety.csv")
