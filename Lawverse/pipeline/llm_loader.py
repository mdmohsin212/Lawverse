import os
from langchain_openai import ChatOpenAI
from typing import Optional, List, Any
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.language_models.llms import LLM
from openai import OpenAI
import json
import os


llm = ChatOpenAI(base_url='https://lightning.ai/api/v1/', api_key=os.getenv("API_KEY"), model="google/gemini-2.5-flash")