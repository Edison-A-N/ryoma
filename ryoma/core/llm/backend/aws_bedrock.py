from typing import List, Dict, Generator, Optional
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage

from ryoma.core.config import settings
from ryoma.core.llm.base import BaseLLM


class BedrockLLM(BaseLLM):
    def __init__(self, model_id: str = "amazon.titan-text-premier-v1:0", **kwargs):
        bedrock_endpoint = settings.AWS_BEDROCK_ENDPOINT or None

        self.chat_model = ChatBedrock(
            model_id=model_id, streaming=False, endpoint_url=bedrock_endpoint
        )
        self.stream_model = ChatBedrock(
            model_id=model_id, streaming=True, endpoint_url=bedrock_endpoint
        )

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
