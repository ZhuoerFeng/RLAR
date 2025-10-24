from glm_api_request.model import GateWays
from agent_prompts import TASK_CLS_PROMPT, RERANK_PROMPT, LLMRM_IMPLEMENT_CODE, LIST_TASK_PROMPT, WRITE_CODE_PROPMT
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from tqdm import tqdm
import re
import serpapi
# from serpapi import GoogleSearch
from serpapi import SerpResults


from huggingface_hub import HfApi
import json, os, shutil, sys

serpapi_client = serpapi.Client(api_key="")



def search_serper_engine(query):
    params = {
        "engine": "google",
        "q": query,
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "num": "10",
        "start": "0",
        "safe": "active",
        "api_key": ""
    }

    try:
        # search = GoogleSearch(params)
        results = serpapi_client.search(params)
        return results
    except Exception as e:
        print("Search failed:", e)
        return None


model = GateWays(model_name="gpt-4.1")
extract_pattern = re.compile(r"####(.*)", re.DOTALL)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(1, 3))
def call_api(message):
    result = model.get_api_result(messages=message, temperature=0.7)
    print(result.content)
    return result.content


def call_task_classifier(item):
    q = item["extra_info"]["question"]
    a = item["extra_info"]["answer"]

    prompt = TASK_CLS_PROMPT + "question: " + q + "\n\n" + "answer: " + a + "\n\n"

    message = [
        {"role": "user", "content": prompt}
    ]

    try:    
        result = call_api(message)
        match = re.search(extract_pattern, result)
        print(match)
        if match:
            result = match.group(1).strip()
        else:
            result = ""
    except Exception as e:
        print(e)
        result = ""
    
    return result


def parse_and_rerank(result_html):
    def parse_google_results(result_html):
        try:
            organic_results = result_html.get("organic_results", [])
            parsed_results = []
            for item in organic_results:
                position = item.get("position", "")
                title = item.get("title", "")
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                related_pages_links = item.get("related_pages_links", [])

                parsed_results.append({
                    "position": position,
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "related_pages_links": related_pages_links
                })
            return parsed_results
        except Exception as e:
            print("Parsing failed:", e)
            return []


    parsed_results = parse_google_results(result_html)
    print(json.dumps(parsed_results, indent=2))


    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": RERANK_PROMPT.format(results=json.dumps(parsed_results, indent=2))}
    ]   

    response = model.get_api_result(messages=messages, temperature=0.1, max_tokens=1000)
    print(response)


    if response.content is not None:
        try:
            pattern = r'\[.*?\]'
            match = re.search(pattern, response.content)
            if match:
                filtered_positions = json.loads(match.group())  
            else:
                filtered_positions = []
        except json.JSONDecodeError as e:
            print("JSON decoding failed:", e)
            filtered_positions = []
    else:
        filtered_positions = []
        print("No content in response.")


    filtered_positions.sort(reverse=False)
    print("Filtered positions:", filtered_positions)

    target_model_id = ""

    for item in parsed_results:
        if item["position"] in filtered_positions:
            target_model_id = item["title"]
            break

    print("Target model id:", target_model_id)

    return target_model_id



def download_huggingface_model(model_id, save_dir='downloaded_models'):
    api = HfApi()
    model_info = api.model_info(model_id)
    print(f"Model info for {model_id}: {model_info}")

    model_dir = os.path.join(save_dir, model_id.replace('/', '_'))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    api.hf_hub_download(
        repo_id=model.modelId, 
        # filename="README.md", 
        repo_type="model",
        local_dir=model_dir
    )
    # model_name_list.append(model.modelId)
    # print('-' * 80)

    print(f"Model {model_id} downloaded to {model_dir}")
    return model_dir


def deploy_reward_model_inference_code_from_readme(save_dir):
    # find readme or demo code
    readme_path = os.path.join(save_dir, "README.md")
    with open(readme_path, 'r') as f:
        readme_content = f.read()

    # open readme and find the target demo code
    
    prompt_to_llm = LLMRM_IMPLEMENT_CODE.format(readme=readme_content, model_path=save_dir)
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt_to_llm}
    ]

    response = model.get_api_result(messages=messages, temperature=0.1, max_tokens=2000)
    print(response)

    code_extract_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    try:
        match = re.search(code_extract_pattern, response.content)
        print(match)
        if match:
            code_content = match.group(1).strip()
        else:
            code_content = ""
    except Exception as e:
        print(e)
        code_content = ""
        
    # output the inference code
    output_code_path = os.path.join(save_dir, "inference_code.py")
    with open(output_code_path, 'w') as f:
        f.write(code_content)
    
    return code_content


def generate_rule_metric_reward_tool(task_cls):
    # plan which metric to implement
    prompt = LIST_TASK_PROMPT.format(task_cls)
    message = [
        {"role": "user", "content": prompt}
    ]

    result = call_api(message)
    print(result)

    metric_extract_pattern = re.compile(r'####(.*)', re.DOTALL)
    try:
        match = re.findall(metric_extract_pattern, result)
        print(match)
        if match:
            metric_name = [x.strip() for x in match]
        else:
            metric_name = []

    except Exception as e:
        print(e)
        metric_name = []

    # generate the code for the metric

    metric_codes = []
    for name in metric_name:
        prompt = WRITE_CODE_PROPMT + name + '\n\n   '
        message = [{"role": "user", "content": prompt}]
        try:
            response = call_api(message)
        except Exception as e:
            print(f"API call failed for metric '{name}':", e)
            response = ""

        # extract python code block if present
        code_extract_pattern = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
        try:
            match = re.search(code_extract_pattern, response)
            if match:
                code_content = match.group(1).strip()
            else:
                code_content = response.strip()
        except Exception as e:
            print("Code extraction failed:", e)
            code_content = response.strip() if response else ""

        # persist generated metric code to disk
        save_dir = "generated_metrics"
        os.makedirs(save_dir, exist_ok=True)
        safe_name = re.sub(r'[^0-9a-zA-Z_]+', '_', name).strip('_')
        file_name = f"{safe_name}_metric.py" if safe_name else "metric.py"
        file_path = os.path.join(save_dir, file_name)
        try:
            with open(file_path, "w") as f:
                f.write(code_content)
            print(f"Wrote metric '{name}' to {file_path}")
        except Exception as e:
            print(f"Failed to write metric file for '{name}':", e)

        metric_codes.append({
            "name": name,
            "code": code_content,
            "path": file_path
        })

    return metric_codes


def wrap_up_tool_descriptions(code_agent_tool_list, web_agent_tool_list):
    tools = []
    for m in code_agent_tool_list:
        tool = {
            "type": "function",
            "function": {
                "name": f"run_metric_{m['name']}",
                "description": (
                    f"Execute the metric '{m['name']}'. "
                    f"Source file path: {m['path']}. "
                    f"This metric implementation:\n{m['code']}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_data": {
                            "type": "string",
                            "description": (
                                "The input dataset or evaluation results to be passed "
                                "to the metric for computation, in a serialized format."
                            )
                        }
                    },
                    "required": ["input_data"]
                }
            }
        }
        tools.append(tool)
    for m in web_agent_tool_list:
        tool = {
            "type": "function",
            "function": {
                "name": f"run_rewardmodel_{m['name']}",
                "description": (
                    f"Execute the web tool '{m['name']}'. "
                    f"Source file path: {m['path']}. "
                    f"This web tool implementation:\n{m['code']}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_data": {
                            "type": "string",
                            "description": (
                                "The input data required by the web tool, in a serialized format."
                            )
                        }
                    },
                    "required": ["input_data"]
                }
            }
        }
        tools.append(tool)
    return tools




def main_workflow():
    item = {
        "extra_info": {
            "question": "Please generate a short story about a brave knight.",
            "answer": "Once upon a time, in a land far away, there lived a brave knight named Sir Lancelot..."
        }
    }

    cls_result = call_task_classifier(item)

    ## Web Agent

    search_engine_results = search_serper_engine(f'best Reward Model for {cls_result} site:huggingface.co')

    target_model_id = parse_and_rerank(search_engine_results)

    save_model_dir = download_huggingface_model(target_model_id)

    code_content = deploy_reward_model_inference_code_from_readme(save_model_dir)     # deploy inference code via README

    # record the process

    ## Code Agent

    generate_rule_metric_reward_tool(cls_result)
    
    # wrap up the tool description

    # call the generated tools


    

    
