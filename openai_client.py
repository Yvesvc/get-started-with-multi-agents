
from langchain_openai import AzureChatOpenAI
from os import environ as env

def get_openai_client():
    return AzureChatOpenAI(
            azure_endpoint = env.get("AZURE_OPENAI_ENDPOINT"),
            openai_api_version = "2024-09-01-preview",
            deployment_name = env.get("AZURE_OPENAI_MODEL"),
            openai_api_key = env.get("AZURE_OPENAI_KEY"),
            openai_api_type = "azure",
            temperature = 0
        )