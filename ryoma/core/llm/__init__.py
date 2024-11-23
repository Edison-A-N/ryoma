from typing import Dict, Any, Literal
from ryoma.core.logging import logger
from ryoma.core.llm.base import BaseLLM


def create_llm(
    provider: Literal["aws_bedrock", "azure_openai", "google_gemini"] = "aws_bedrock",
    model_id: str = "amazon.titan-text-premier-v1:0",
    **kwargs: Dict[str, Any],
) -> BaseLLM:
    """Create LLM instance based on provider

    Args:
        provider: LLM provider, supports 'aws_bedrock', 'azure_openai' and 'google_gemini'
        model_id: Model identifier
        **kwargs: Additional configuration for the LLM

    Returns:
        BaseLLM instance
    """
    if provider == "aws_bedrock":
        from ryoma.core.llm.backend.aws_bedrock import BedrockLLM
        return BedrockLLM(model_id=model_id, **kwargs)

    if provider == "azure_openai":
        from ryoma.core.llm.backend.azure_openai import AzureOpenAILLM

        return AzureOpenAILLM(model_id=model_id)

    if provider == "google_gemini":
        from ryoma.core.llm.backend.google_gemini import GeminiLLM

        return GeminiLLM(model_id=model_id)

    raise ValueError(f"Unsupported provider: {provider}")
