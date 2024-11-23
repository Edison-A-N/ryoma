from typing import List, Dict, Generator, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from ryoma.core.config import settings
from ryoma.core.llm.base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model_id: str):
        self.model = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.GOOGLE_API_KEY,
            convert_system_message_to_human=True,
        )
        self.history = []

    def chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        if history:
            self.history = self._convert_history(history)
        else:
            self.history = []

        self.history.append(HumanMessage(content=prompt))
        response = self.model.invoke(self.history)
        self.history.append(response)
        return response.content

    def stream_chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> Generator[str, None, None]:
        if history:
            self.history = self._convert_history(history)
        else:
            self.history = []

        self.history.append(HumanMessage(content=prompt))
        response = self.model.stream(self.history)
        for chunk in response:
            if chunk.content:
                yield chunk.content

    def _convert_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        converted = []
        for msg in history:
            if msg["role"] == "user":
                converted.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                converted.append(AIMessage(content=msg["content"]))
        return converted
