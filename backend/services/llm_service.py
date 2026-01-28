import numpy as np
from typing import List, Union
from openai import OpenAI
from huggingface_hub import InferenceClient
from config import HF_API_TOKEN


class LLMService:
    """Handles AI interactions with LLM."""

    def __init__(self):
        self.chat_model = "Qwen/Qwen2.5-7B-Instruct"
        self.embedding_model = "BAAI/bge-small-en-v1.5"
        self.chat_client = OpenAI(
            api_key=HF_API_TOKEN,
            base_url="https://router.huggingface.co/v1"
        )
        self.inference_client = InferenceClient(api_key=HF_API_TOKEN)

    def get_answer(self, question, context) -> str:
        """
        Generates a natural language response based on the retrieved context.

        Args:
            question (str): The original user inquiry.
            context (str): The most relevant text chunk retrieved from the vector database.

        Returns:
            str: The AI-generated response.
        """
        try:
            response = self.chat_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": f"You are a hotel assistant. "
                                                  f"Use this context to answer the guest: {context}"},
                    {"role": "user", "content": question}
                ],
                max_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Error processing request: {str(e)}"

    def embed_content(self, content: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """
        Generates embeddings for the given content.

        Args:
            content (str): The text to embed.
            is_query (bool): If True, applies the 'query' prefix for search requests.
            If False, applies the 'passage' prefix for indexing documents. Defaults to False.

        Returns:
            np.ndarray: A float32 NumPy array containing the generated embeddings.
        """
        prefix = "query: " if is_query else "passage: "

        input_data = [content] if isinstance(content, str) else content
        input_with_prefix = [f"{prefix}{text}" for text in input_data]

        try:
            embeddings = self.inference_client.feature_extraction(
                input_with_prefix,
                model=self.embedding_model
            )
            return np.array(embeddings).astype('float32')
        except Exception as e:
            print(f"⚠️ Embedding Error: {str(e)}")
            return np.array([])
