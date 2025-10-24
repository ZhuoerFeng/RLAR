TASK_CLS_PROMPT = """
You will be given a `question` and an `answer` from a language model interaction.  
Your job is to determine the main type of task being performed. Examples include but are not limited to: translation, summarization, question answering, code generation, math solving, creative writing, RLHF alignment, explanation, classification, etc.  
You do not need to list all possible task types — instead, use your judgment to give the most fitting label for this specific instance.  
Rules:  
1. Output only the task type as a concise label (maximum **three words**).  
2. Do not include any extra text, punctuation, or explanation.  
3. If uncertain, choose the closest-fitting description.  

Provide your output results after four sharp marks ####, such as "#### Translation".

**Input example:**  
```json
"question": "You are a helpful assistant that translates English sentences to French. Following the below input-output format\n[English Input]\nxxxx\n[French Output]\nxxxx\nStart your translation task now. [English Input]\nMichael Gill formed The Murder Mile with his old friend, ex- Spring Heeled Jack USA and Lost City Angels frontman, Ron Ragona.\n[French Output]\n",
"answer": "Michael Gill a formé The Murder Mile avec son vieil ami, l'ancien chanteur de Spring Heeled Jack USA et Lost City Angels, Ron Ragona."
```
**Output:**  
#### English–French Translation

Now classify the given instance according to these rules.

"""


TASK_DECOMP_PROMPT = """Please break down the following generative task into a combination of several basic generative tasks:  

Basic task list:
1. Controlled generation: Generate coherent natural language text that meets certain given conditions. Best for simple, clear tasks; complex writing should be split into smaller steps like planning and cloze generation.
2. Translation: Generate a corresponding text in another natural language from a text in one natural language.  
3. Text summarization: Summarize the given text, retaining the main information.  
4. Question answering: Provide appropriate answers based on background information and question requests provided by the user.  
5. Paraphrasing: Modify the provided text into a different form of expression that meets the given rewriting requirements.  
6. Cloze generation: Given a continuous piece of text with missing parts, generate appropriate text for the missing positions so that the original text becomes complete, coherent, and consistent.  
7. Planning generation: Plan a high-level outline in order to accomplish a relatively complex generative task, such as creating a chapter list, designing character traits, designing scripts, or designing a timeline.  
8. Code: Generate executable code that meets the specified requirements, or supplement or revise code according to the given requirements. The defining criterion for this task is that the output is primarily code.


Decomposition goal:
- Break down the complex generative task provided by the user into a list composed of the above basic tasks according to its logical steps.  
- Steps should be arranged in execution order, and the description should start from the original input form and proceed until the task is completed.  
- Each step must clearly specify the “basic task type” and the execution content of that step.  
- If the task does not need to be broken down, provide a single-step basic task and rewrite its description into a clearer instruction that aligns with the type of task in the basic task list.


Output format requirements:
- List the decomposition results step-by-step (step number + basic task type name + specific execution description).  
- Enclose the final result within `<Result> ... </Result>` tags.  

Below is an example:  
[Example Start]  
Task to be decomposed: Please provide an English summary for the following Chinese document.
Decomposition result:  
1. Translation: Please translate the following Chinese document into an English document.  
2. Text summarization: Please summarize the given English document, and ensure the summary does not exceed 200 words.  
[Example End]  

**Now,** perform the above decomposition process on the given question (or task description) below, and write the final decomposition result within `<Result> ... </Result>` tags.


{original_task}
"""


LIST_TASK_PROMPT = """You are an expert in designing reward models and evaluation metrics for the **{task}** task.  
Your goal is to list **3–5 possible reward model or evaluation metric choices** for this task, drawing from the following two categories:  

1. **Rule-based** – Explicit rules (e.g., exact match with reference output, length constraints) used directly as rewards.  
2. **Metric-based** – Standard NLP metrics (e.g., BLEU, ROUGE, METEOR) used to evaluate and reward generated results.  

**Output formatting requirements:**  
- Place your results **after four hash marks (`####`)**.  
- For **each choice**, indicate its **category** and **name**, using the format:  
  ```
  #### <Category>/<Name>: <Brief description>
  ```  
- Use a **new line** for each choice.  

**Example:**  
```
#### Metric-based/BLEU: Measures the n-gram overlap between generated output and reference text.
#### Rule-based/Length: Rewards outputs within the target length range for conciseness.
```

"""


WRITE_CODE_PROPMT = """Implement the following metric according to description using python. You are free to use packages. You should write a function begin with 'compute_xxx' where xxx is the name of the metric. The function accepts:
- prompt: the instruction to the prompt
- candidate_response: the candidate response to be evaluated by the metric
- reference_response: the reference answer for the prompt
You should directly return a scaler score.

Output the python code in ```python\n xxxx\n```. And list the requirements within `````` use requirements.txt style. 
  
[metric description]

"""


WRAP_TOOL_PROMPT = """Write a short description of the following reward function, based on the name, python code implementation. In your description, briefly explain what the function calculates and how it can be used to evaluate text generation quality. The description should be concise and clear, suitable for someone familiar with NLP evaluation metrics.

NAME
{name}

CODE

```python
{code}
```

write your description after four sharp marks ####, such as "#### This function calculates ...".
"""



METRIC_DESIGN_PROMPT = """You are an expert in designing evaluation metrics writing with python code. Currently you nedd to implement the {metric}. The introduction for the metric is as follows:

{info}

Suppose the input is prompt, candidate response, reference response. write the metric as a python function that receive the above parameter, and return with the score for it. Name it begin with “compute_”, such as “def compute_XXX(...)“

Make sure the code is correct and can be run without error. Include the script within ```python ... ```.

"""


REWARD_MODEL_DESIGN_PROMPT = """You are an expert in launching reward models with python code. Write a script that supports calculating the reward score of some text. You should write a function, that support input parameter:

- prompt: str, instruction or context conditions
    
- response: str, the text need to be evaluated
    
- reference: str, some reference answer/response for the above prompt
    
You are given the following README.md of one reward model, and the local model checkpoint path. The cuda device for the model is “cuda:0“. Name the calculation function starting with "compute_", such as "def compute_XXX(...)". Make sure the model  checkpoint is loaded precisely once in the script.

README.md:

{readme}


local model checkpoint path: {model_path}
"""


RERANK_PROMPT = """You are given a list of search engine results with position IDs.  
Your task is to filter them according to the following rules:

1. **Identify Reward Models:**  
   - Keep only results that are **reward model** links.  
   - Reward models often have model names containing keywords like `-Reward-` or `-RM-`.  
   - Discard results for base models (`-Base`) or instruct models (`-Instruct`) or chat models (`-Chat`).  
   - If a model name has none of these hints, and it’s unclear whether it is a reward model, discard it.

2. **Hugging Face Model Repositories Only:**  
   - Keep only links pointing to **Hugging Face model repositories**.  
   - Discard datasets, research papers, blog posts, or other non-model content.


3. **Score Output Format only:**
   - Regression models only, in other words, models that output a score (e.g., 0-1) rather than generating text.

Directly discard those items that violates rule 1, 2 or 3 and keep the rest items. Output the resting items in list using their original position id like "[0, 1, 3, 5, ...]". If none of the items are left, output an empty list "[]".

{results}
"""


LLMRM_IMPLEMENT_CODE = """Implement a python script for launching a reward model according to the following README.md and local model checkpoint path. The cuda device for the model is "cuda:0". You should write a function, that support input parameter:
- prompt: str, instruction or context conditions
- response: str, the text need to be evaluated
- reference: str, some reference answer/response for the above prompt

Your implementation are free to use the packages mentioned in the README.md. Name the calculation function starting with "compute_", such as "def compute_XXX(...)" where XXX should be the reward model name or related abbreviation. Make sure the model checkpoint is loaded precisely once in the script. Format your output enclosed within ```python\n xxxx\n```.

[readme]

{readme}

[your implementation]

"""


