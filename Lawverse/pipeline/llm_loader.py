from typing import Optional, List
from langchain_core.language_models.llms import LLM
from openai import OpenAI
import os

# env = environ.Env()
# environ.Env.read_env(Path(__file__).resolve().parent.parent.parent / ".env")

class LightningLLM(LLM):
    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "client", OpenAI(base_url="https://lightning.ai/api/v1/", api_key=api_key))
        object.__setattr__(self, "model", model)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            temperature=0.1,
            max_tokens=5120
        )
        return resp.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "lightning_gpt"

llm = LightningLLM(api_key=os.getenv("API_KEY"), model=os.getenv("MODEL"))