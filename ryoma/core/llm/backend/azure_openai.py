from typing import List, Dict, Generator, Optional
from langchain_openai import AzureChatOpenAI
from ryoma.core.config import settings
from ryoma.core.llm.base import BaseLLM


class AzureOpenAILLM(BaseLLM):
    def __init__(self, model_id: str):
        self.client = AzureChatOpenAI(
            openai_api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            openai_api_version="2024-06-01",
            model_name=model_id,
            streaming=True,
        )
        self.model_id = model_id

    def chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        messages = self._build_messages(prompt, history)
        response = self.client.invoke(messages)
        return response.content

    def stream_chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, history)
        response = self.client.stream(messages)
        for chunk in response:
            if chunk.content:
                yield chunk.content

    def _build_messages(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages
