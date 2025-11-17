import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url='https://lightning.ai/api/v1/', api_key=os.getenv("API_KEY"), model="google/gemini-2.5-flash",)