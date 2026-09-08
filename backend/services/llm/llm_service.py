import json
import os
import yaml
import numpy as np
from typing import List, Union
from openai import OpenAI
from huggingface_hub import InferenceClient
from backend.constants import LLMRole
from config import HF_API_TOKEN, LLM_BASE_URL, LLM_API_KEY, CHAT_MODEL, JUDGE_MODEL, EMBEDDING_MODEL


class LLMService:
    """Handles AI interactions with LLM."""

    def __init__(self, role: LLMRole = LLMRole.ASSISTANT):
        ROLE_MODEL_MAP = {
            LLMRole.ASSISTANT: CHAT_MODEL,
            LLMRole.JUDGE: JUDGE_MODEL
        }
        self.chat_model = ROLE_MODEL_MAP[role]
        self.embedding_model = EMBEDDING_MODEL
        self.chat_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )
        self.inference_client = InferenceClient(api_key=HF_API_TOKEN)
        self.role = role

    def _get_system_prompt(self, role: LLMRole) -> str:
        """Gets the system prompt for the specified role."""
        ROLE_PROMPT_MAP = {
            LLMRole.ASSISTANT: "hotel_chat_assistant.yaml",
            LLMRole.JUDGE: "llm_as_a_judge.yaml"
        }

        file_path = os.path.join(os.path.dirname(__file__), "prompts", ROLE_PROMPT_MAP[role])

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            prompt = data["system_prompt"]
            return prompt

    def get_answer(self, query: str, context: list[str]) -> dict:
        """
        Generates a natural language response based on the retrieved context.

        Args:
            query (str): The original user inquiry.
            context (list[str]): The most relevant text chunks retrieved from the vector database.

        Returns:
            dict: {"confidence": bool, "answer": str}, on both the success and
            failure paths, so callers can rely on a single dict-shaped contract.
        """
        system_prompt = self._get_system_prompt(self.role)
        context = "\n".join(context)
        prompt = f"{system_prompt}\n\nCONTEXT:\n'''\n{context}\n'''"
        try:
            create_kwargs = {
                "model": self.chat_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 150,
                "temperature": 0.1
            }
            if self.role == LLMRole.JUDGE:
                # Some Judge models (e.g. Qwen3) default to a "thinking" mode
                # that spends the token budget on hidden reasoning before any
                # visible output. Disabling it keeps latency and output length
                # comparable to non-thinking models under the same max_tokens.
                create_kwargs["extra_body"] = {"reasoning_effort": "none"}

            response = self.chat_client.chat.completions.create(**create_kwargs).choices[0].message.content

            return json.loads(response)
        except Exception as e:
            return {"confidence": False, "answer": f"⚠️ Error processing request: {str(e)}"}

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
