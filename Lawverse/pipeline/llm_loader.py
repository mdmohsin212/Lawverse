import os
from langchain_openai import ChatOpenAI
from typing import Optional, List, Any
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.language_models.llms import LLM
from openai import OpenAI
import json
import os

# class LightningLLM(LLM):
#     def __init__(self, model: str, **kwargs):
#         super().__init__(**kwargs)
#         object.__setattr__(self, "client", OpenAI(base_url="https://lightning.ai/api/v1/", api_key=os.getenv("API_KEY")))
#         object.__setattr__(self, "model", model)

#     def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional = None, **kwargs: Any) -> str:
        
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
#             temperature=0.1,
#             max_completion_tokens=2048
#         )
#         return resp.choices[0].message.content

#     def with_structured_output(self, schema: type[BaseModel]):
#         def structured_call(user_input: str):
#             schema_json = json.dumps(schema.model_json_schema(), indent=2)
            
#             system_instruction = f"""
#             You MUST output only valid JSON that strictly matches this schema:
#             {schema_json}
#             Rules:
#             - Do NOT add explanations.
#             - Do NOT add additional fields.
#             - Output ONLY JSON.
#             """
            
#             full_prompt = f"{system_instruction}\nUser Query:\n{user_input}"
#             raw_output = self._call(full_prompt)
            
#             try:
#                 data = json.loads(raw_output)
#             except:
#                 try:
#                     json_part = raw_output[raw_output.find("{"):raw_output.rfind("}") + 1]
#                     data = json.loads(json_part)
#                 except Exception as e:
#                     raise ValueError(f"Model did not return valid JSON:\n{raw_output}") from e

#             return schema(**data)
#         return structured_call
    
#     @property
#     def _identifying_params(self):
#         return {"model": self.model}

#     @property
#     def _llm_type(self):
#         return "lightning_gpt"

# llm = LightningLLM(model='google/gemini-2.5-flash')
llm = ChatOpenAI(base_url='https://lightning.ai/api/v1/', api_key=os.getenv("API_KEY"), model="google/gemini-2.5-flash")