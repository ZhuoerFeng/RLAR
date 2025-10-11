# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code
import json
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
import re

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    enclosure_weight = 1.0
    
    original_solution_str = solution_str
    if '</think>' in solution_str:
        solution_str = solution_str.split('</think>')[-1].strip()
    

    exp_name = extra_info.get("exp_name", "") if extra_info is not None else ""

    if exp_name == 'RLVR':
        target_route = "skywork_llama"

        BASE_URL = 'http://172.18.90.44:5098'
        url = '{}/{}'.format(BASE_URL, target_route)

        prompt = extra_info.get("question", "")
        payloads = {"prompt": prompt, "response": solution_str, "ground_truth": ground_truth}
        @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
        def call_judge_model_api(payloads): 
            myrequest = requests.post(
                url=url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payloads).encode('utf-8')
            )
            if myrequest.status_code != 200:
                raise Exception(f"API call failed with status code {myrequest.status_code}: {myrequest.text}")
            
            return myrequest.json()['score']
        try:
            res = call_judge_model_api(payloads)
        except Exception as e:  
            print(f"Error calling judge model API: {e}")
            res = -20.0
        enclosure_weight = 10.0


    elif exp_name == 'RLVR':
        call_function = extra_info.get("function_call", None)
        target_route = ""
        if call_function is not None:
            target_route = call_function
        else:
            print("No call_function found, use skywork_llama as default")
            target_route = "skywork_llama"
                

        BASE_URL = 'http://172.18.90.44:5098'
        url = '{}/{}'.format(BASE_URL, target_route)

        prompt = extra_info.get("question", "")
        payloads = {"prompt": prompt, "response": solution_str, "ground_truth": ground_truth}
        @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
        def call_judge_model_api(payloads): 
            myrequest = requests.post(
                url=url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payloads).encode('utf-8')
            )
            if myrequest.status_code != 200:
                raise Exception(f"API call failed with status code {myrequest.status_code}: {myrequest.text}")
            
            return myrequest.json()['score']
        try:
            res = call_judge_model_api(payloads)
        except Exception as e:  
            print(f"Error calling judge model API: {e}")
            res = -20.0
        enclosure_weight = 10.0
        

    ## WITH RULE/METRIC ensemble
    elif exp_name == 'lazyrule':
        BASE_URL = 'http://172.18.90.44:5098'
        prompt = extra_info.get("question", "")
        payloads = {"prompt": prompt, "response": solution_str, "ground_truth": ground_truth}
        @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
        def call_judge_model_api(payloads):
            url1 = '{}/bleu1'.format(BASE_URL)
            url2 = '{}/length'.format(BASE_URL)
            myrequest = requests.post(
                url=url1,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payloads).encode('utf-8')
            )
            if myrequest.status_code != 200:
                raise Exception(f"API call failed with status code {myrequest.status_code}: {myrequest.text}")
            
            s1 = myrequest.json()['score']

            myrequest = requests.post(
                url=url2,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payloads).encode('utf-8')
            )
            if myrequest.status_code != 200:
                raise Exception(f"API call failed with status code {myrequest.status_code}: {myrequest.text}")
            s2 = myrequest.json()['score']

            return 0.7 * s1 + 0.3 * s2
        
        if "gsm8k" in data_source:
            from . import gsm8k
            res = gsm8k.compute_score(solution_str, ground_truth)
        else:
            try:
                res = call_judge_model_api(payloads)
            except Exception as e:
                print(f"Error calling judge model API: {e}")
                res = 0
        print(res)
        enclosure_weight = 1.0

    elif data_source in ['zhuoer/howto', 'zhuoer/abs2text']:
        from . import infilling
        prompt = extra_info.get("question", "")
        split = extra_info.get("split", "")
        # res = infilling.compute_bleu_score(solution_str, ground_truth)
        if split == "train":
            res = infilling.reward_ensemble(prompt, solution_str, ground_truth)
        else:
            res = (infilling.compute_bleu_score(solution_str, ground_truth) + infilling.compute_length_penalty(solution_str, ground_truth)) / 2
            
    elif data_source in ['zhuoer/howto_val', 'zhuoer/english-french-translation_en-fr', 'zhuoer/english-french-translation_fr-en']:
        
        from . import infilling
        res = infilling.compute_bleu_score(solution_str, ground_truth)
    elif data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if '<think>' in original_solution_str and '</think>' in original_solution_str:
        tmp = original_solution_str.split('</think>')[0].strip()
        if '<think>' in tmp:
            res += enclosure_weight

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
