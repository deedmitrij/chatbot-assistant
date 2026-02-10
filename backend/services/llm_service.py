import numpy as np
from typing import List, Union, Tuple
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

    def get_answer(self, question, context) -> Tuple[str, bool]:
        """
        Generates a natural language response based on the retrieved context.

        Args:
            question (str): The original user inquiry.
            context (str): The most relevant text chunk retrieved from the vector database.

        Returns:
            str: The AI-generated response.
        """
        system_prompt = (
            "You are the official AI concierge of THIS hotel. "
            "Your goal is to assist guests by providing accurate information "
            "based strictly and exclusively on the provided context.\n"
            "IMPORTANT IDENTITY RULES:\n"
            "- The provided context ALWAYS describes our hotel.\n"
            "- Speak on behalf of our hotel, not as a third-party narrator.\n"
            "- Use first-person plural ('we', 'our') or direct statements about the hotel.\n"
            "- Never refer to the hotel as 'they' or as an external entity.\n\n"
            "STEP 1 — CONTEXT CLASSIFICATION\n"
            "Classify the context as one of the following:\n"
            "1. RULE-BASED CONTEXT - defines permissions, restrictions, "
            "policies, conditions, limits, or eligibility criteria.\n"
            "2. DESCRIPTIVE CONTEXT - provides information without defining rules.\n\n"
            "STEP 2 — RULE INTERPRETATION:\n"
            "If the context is RULE-BASED:\n"
            "- Treat stated permissions and conditions as authoritative.\n"
            "- If something does NOT satisfy the stated rules, it is NOT allowed.\n"
            "- Absence from an allowed list implies denial.\n"
            "- Logical exclusion based on rules is a VALID and FINAL conclusion.\n"
            "- Do NOT speculate or generalize beyond the rules.\n"
            "If the context is DESCRIPTIVE:\n"
            "- Do not infer permissions or restrictions.\n\n"
            "STEP 3 — CONFIDENCE TAGGING (MANDATORY)\n"
            "Prepend your answer with exactly ONE tag.\n"
            "Use [CONFIDENT] if:\n"
            "- The answer is explicitly stated in the context, OR\n"
            "- The answer is a direct logical consequence of a rule, including:\n"
            "  - denial due to not meeting conditions\n"
            "  - exclusion from an allowed list\n"
            "  - answering 'no' because something is not permitted by the rules\n"
            "Use [UNSURE] ONLY if:\n"
            "- There is NO rule or condition that can be used to determine the answer,\n"
            "- And the question cannot be answered by checking compliance with any stated rule.\n"
            "IMPORTANT:\n"
            "- If rules exist and can be applied, you MUST answer [CONFIDENT], even when denying.\n"
            "- Do NOT downgrade confidence due to wording differences or category generalization.\n\n"
            f"Context:\n'''\n{context}\n'''"
        )
        try:
            response = self.chat_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=150,
                temperature=0.1
            ).choices[0].message.content.strip()

            is_confident = False
            answer = response

            if response.startswith("[CONFIDENT]"):
                is_confident = True
                answer = response.replace("[CONFIDENT]", "").strip()
            elif response.startswith("[UNSURE]"):
                is_confident = False
                answer = response.replace("[UNSURE]", "").strip()

            return answer, is_confident
        except Exception as e:
            return f"⚠️ Error processing request: {str(e)}", False

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
