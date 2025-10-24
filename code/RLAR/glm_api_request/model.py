import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
    
    
class GateWays:
    def __init__(self, model_name):
        self.model = model_name
        self.api_url = ""
        self.api_key = ""

        self.client = OpenAI(base_url=self.api_url, api_key=self.api_key)
        

    def get_api_result(self, messages:list, tools: list = None, temperature: float = 1.0, max_tokens: int = 4000):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            # max_tokens=max_tokens,
            temperature=temperature,
            timeout=30
        )
        # print(response)
        return response.choices[0].message
    


