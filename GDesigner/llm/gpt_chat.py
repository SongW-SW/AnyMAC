import aiohttp
from typing import List, Union, Optional
import traceback
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from typing import Dict, Any
from dotenv import load_dotenv
import os
import json

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry


OPENAI_API_KEY = os.getenv('API_KEY', 'your_openai_api_key_here')
BASE_URL = 'https://api.openai.com/v1'

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL') or BASE_URL
MINE_API_KEY = os.getenv('API_KEY') or OPENAI_API_KEY

class APIError(Exception):
    """Exception raised for errors in the API request."""
    def __init__(self, status, message, response_text=None):
        self.status = status
        self.message = message
        self.response_text = response_text
        super().__init__(f"API Error: {status} - {message}")

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((aiohttp.ClientError, APIError, json.JSONDecodeError))
)
async def achat(
    model: str,
    msg: List[Dict],):
    # Use the OpenAI chat completions API endpoint
    request_url = f"{BASE_URL}/chat/completions"
    authorization_key = f"Bearer {OPENAI_API_KEY}"
    
    if not OPENAI_API_KEY:
        raise ValueError("API_KEY environment variable is not set. Please set it in your .env file or environment variables.")
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': authorization_key
    }
    
    # Format messages in the way OpenAI API expects
    formatted_messages = []
    for m in msg:
        if isinstance(m, dict) and 'role' in m and 'content' in m:
            formatted_messages.append({"role": m['role'], "content": m['content']})
        elif isinstance(m, Message):
            formatted_messages.append({"role": m.role, "content": m.content})
    # import ipdb; ipdb.set_trace()
    data = {
        "model": model,
        "messages": formatted_messages,
        "temperature": 0.0,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(request_url, headers=headers, json=data) as response:
                response_text = await response.text()
                
                if response.status >= 400:
                    print(f"API Error: {response.status} - {response_text}")
                    raise APIError(
                        status=response.status,
                        message=f"API request failed with status {response.status}",
                        response_text=response_text
                    )
                
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise APIError(
                        status=response.status,
                        message=f"Failed to parse API response as JSON: {str(e)}",
                        response_text=response_text
                    )
                
                if 'choices' not in response_data:
                    raise APIError(
                        status=response.status,
                        message="API response missing 'choices' field",
                        response_text=str(response_data)
                    )
                
                # Extract the response text from the OpenAI API response
                response_text = response_data['choices'][0]['message']['content']
                
                prompt = "".join([item['content'] for item in msg])
                cost_count(prompt, response_text, model)
                return response_text
    except aiohttp.ClientError as e:
        print(f"Network error during API request: {str(e)}")
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"Unexpected error during API request: {str(e)}")
        traceback.print_exc()
        raise

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

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
        
        try:
            return await achat(self.model_name, messages)
        except Exception as e:
            print(f"Error in agen method: {str(e)}")
            traceback.print_exc()
            raise
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass