from typing import List, Dict, Generator, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from ryoma.core.config import settings
from ryoma.core.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model_id: str):
        self._model_id = model_id
        self._chat_model = ChatOpenAI(
            model_name=model_id,
            openai_api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_ENDPOINT,
            streaming=False,
        )
        self.stream_model = ChatOpenAI(
            model_name=model_id,
            openai_api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_ENDPOINT,
            streaming=True,
        )

    @property
    def chat_model(self) -> ChatOpenAI:
        return self._chat_model

    def _convert_history(self, history: Optional[List[Dict[str, str]]]) -> List:
        if not history:
            return []
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        messages = self._convert_history(history)
        messages.append(HumanMessage(content=prompt))
        response = self.chat_model.invoke(messages, **kwargs)
        return response.content

    def stream_chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> Generator[str, None, None]:
        messages = self._convert_history(history)
        messages.append(HumanMessage(content=prompt))

        for chunk in self.stream_model.stream(messages, **kwargs):
            if chunk.content:
                yield chunk.content

    def get_provider(self) -> str:
        return "openai"

    def get_model_id(self) -> str:
        return self._model_id
