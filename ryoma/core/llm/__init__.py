from typing import Dict, Any, Literal
from .base import BaseLLM


def create_llm(
    provider: Literal["aws_bedrock"] = "aws_bedrock",
    model_id: str = "amazon.titan-text-premier-v1:0",
    **kwargs: Dict[str, Any],
) -> BaseLLM:
    """Create LLM instance based on provider

    Args:
        provider: LLM provider, currently only supports 'aws_bedrock'
        model: Model identifier
        **kwargs: Additional configuration for the LLM

    Returns:
        BaseLLM instance
    """
    if provider == "aws_bedrock":
        from ryoma.core.llm.backend.aws_bedrock import BedrockLLM

        return BedrockLLM(model_id=model_id, **kwargs)

    raise ValueError(f"Unsupported provider: {provider}")
