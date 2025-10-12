import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = 'http://localhost:11434'

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL') or BASE_URL
MINE_API_KEYS = os.getenv('API_KEY') or ''


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict]):
    request_url = f"{MINE_BASE_URL}/api/generate"
    # import ipdb; ipdb.set_trace()
    headers = {
        'Content-Type': 'application/json'
    }

    # 將 msg 轉為字串 prompt
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in msg])

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": 42
        }

    }
    # write promt to a file called prompt.txt if exist using append mode
    # with open('prompt.txt', 'a') as f:
    #     f.write(prompt)
    #     f.flush()

    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")

            response_data = await response.json()
            completion = response_data.get('response', '')
            cost_count(prompt, completion, model)
            # # 儲存response（optional）
            # with open('response.txt', 'a') as f:
            #     f.write(completion + "\n\n")
            #     f.flush()

            return completion


@LLMRegistry.register('SLLMChat')
class SLLMChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        from tenacity import RetryError

        try:
            completion = await achat(self.model_name,messages)
        except RetryError as e:
            print("RetryError happened!")
            print("Last attempt exception:", e.last_attempt.exception())
            # Return an error message instead of undefined completion
            completion = "Error: Failed to get response after multiple attempts."
                    
        return completion
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
