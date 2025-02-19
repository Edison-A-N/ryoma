from typing import Dict, Any, Literal, Optional
from ryoma.core.llm.base import BaseLLM
from ryoma.core.config import settings

def create_llm(
    provider: Optional[
        Literal["aws_bedrock", "openai", "google_gemini", "zhipu"]
    ] = None,
    model_id: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> BaseLLM:
    """Create LLM instance based on provider

    Args:
        provider: LLM provider, supports 'aws_bedrock', 'openai' and 'google_gemini'
        model_id: Model identifier
        **kwargs: Additional configuration for the LLM

    Returns:
        BaseLLM instance
    """
    if provider is None:
        provider = settings.LLM_PROVIDER
        model_id = settings.LLM_MODEL_ID
    else:
        if model_id is None:
            raise ValueError("model_id is required")

    if not provider:
        raise ValueError("provider is required")
    if not model_id:
        raise ValueError(f"model_id is required, got {model_id}")

    if provider == "aws_bedrock":
        from ryoma.core.llm.backend.aws_bedrock import BedrockLLM
        return BedrockLLM(model_id=model_id, **kwargs)

    if provider == "openai":
        from ryoma.core.llm.backend.openai import OpenAILLM

        return OpenAILLM(model_id=model_id)

    if provider == "google_gemini":
        from ryoma.core.llm.backend.google_gemini import GeminiLLM

        return GeminiLLM(model_id=model_id)

    if provider == "zhipu":
        from ryoma.core.llm.backend.zhipu import ZhipuLLM

        return ZhipuLLM(model_id=model_id)

    raise ValueError(f"Unsupported provider: {provider}")
