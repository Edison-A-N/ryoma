from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Optional
from langchain_core.language_models.chat_models import BaseChatModel


class BaseLLM(ABC):
    @property
    @abstractmethod
    def chat_model(self) -> BaseChatModel:
        """The underlying chat model instance"""
        pass

    def get_model(self) -> BaseChatModel:
        """Get the underlying model instance"""
        return self.chat_model

    @abstractmethod
    def chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> str:
        """Single chat completion

        Args:
            prompt: User input text
            history: Chat history in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional parameters passed to the model

        Returns:
            Model response text
        """
        pass

    @abstractmethod
    def stream_chat(
        self, prompt: str, history: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> Generator[str, None, None]:
        """Streaming chat completion

        Args:
            prompt: User input text
            history: Chat history in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional parameters passed to the model

        Yields:
            Model response text chunks
        """
        pass
