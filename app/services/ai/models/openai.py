from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def create_openai_model(*, model_name: str = "gpt-4o-mini") -> BaseChatModel:
    return ChatOpenAI(
        model=model_name,
        use_responses_api=True,
        output_version="responses/v1",
    )
