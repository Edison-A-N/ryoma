from typing import List, Dict, Generator, Optional
from openai import OpenAI
from ryoma.core.config import settings
from ryoma.core.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model_id: str):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_ENDPOINT
        )
        self.model_id = model_id

    def chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        messages = self._build_messages(prompt, history)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content

    def stream_chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> Generator[str, None, None]:
        messages = self._build_messages(prompt, history)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_messages(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages
