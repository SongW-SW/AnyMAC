from typing import Optional
from class_registry import ClassRegistry

from GDesigner.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "gpt-4o"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif model_name in ["gpt-4o", "o3-mini", "gpt-4o-2024-11-20", "gpt-4-1106-preview"]:  # OpenAI models
            model = cls.registry.get('GPTChat', model_name)
        else:  # Small LLMs running on Ollama
            model = cls.registry.get('SLLMChat', model_name)

        return model
