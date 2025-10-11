import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import json


TASK_RM_MAPPING = {
    "Translation": "seed",
    "Controlled generation": "skywork_qwen",
    "Text summarization": "skywork_qwen",
    "Paraphrasing": "gpt2_large_helpful",
    "Cloze generation": "skywork_llama",
    "Question answering": "deberta_reward",
    "Planning generation": "skywork_llama",
    "Code": "skywork_qwen"
}


def calculate_step_score(outer_prompt, task_list, instruction_list, response_list, ground_truth, extra_info):
    step_scores = []

    prefix_prompt = "Judge based on the following instruction and response. If there are multiple instruction/responses, judge the latest instruction and corresponding response based on previous contexts.\n\n"

    BASE_URL = 'http://172.18.90.44:5098'

    data = []

    history_context = ""

    for task, instruction, response in zip(task_list, instruction_list, response_list):
        cur_prompt = history_context.strip() + "\n\n" + '<instruction>\n' + instruction + '\n</instruction>\n'
        data.append({"task": task, "prompt": prefix_prompt + cur_prompt, "response": response})
        history_context +=  "\n\n" + '<instruction>\n' + instruction + '\n</instruction>\n' + '<response>\n' + response + '\n</response>\n'

    def call_judge_model_api(item):
        task = item['task']
        prompt = item['prompt']
        response = item['response']
        # route = TASK_RM_MAPPING.get(task, "Skywork-Reward-V2-Qwen3-8B")
        route = TASK_RM_MAPPING[task]
        url = '{}/{}'.format(BASE_URL, route)
        payloads = {"prompt": prompt, "response": response, "ground_truth": ground_truth}

        # print(json.dumps(payloads, indent=4, ensure_ascii=False))
        print("Calling URL:", url)

        @retry(stop=stop_after_attempt(5), wait=wait_fixed(1))
        def call_api(payloads):
            response = requests.post(url, json=payloads, timeout=30)
            if response.status_code == 200:
                return response.json()['score']
            else:
                raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
            
        if response == "":
            score = -20.0
        else:
            score = call_api(payloads)
        
        if score is None:
            score = 0

        return score
    
    call_wrapper = partial(call_judge_model_api)
    with ThreadPoolExecutor(max_workers=10) as executor:
        for score in executor.map(call_wrapper, data):
            step_scores.append(score)
    
    return step_scores


if __name__ == "__main__":
    outer_prompt = "Translate the following English sentence to French: 'Hello, how are you?'"

    instruction_list = ["Translate the following English sentence to French: 'Hello, how are you?'", "Prolong the above translated French into 200 words, supposing in a classroom scenario.'", "Summarize the above French text into 50 words."]

    response_list = [
        'The translation of "Hello, how are you?" into French is:\nBonjour, comment allez-vous ?',
        """In a classroom scenario, extending the above greeting into approximately 200 words:\n\nBonjour, comment allez-vous ? demanda Monsieur Dupont en entrant dans la salle de classe lumineuse. Les élèves, assis à leurs bureaux, levèrent les yeux et certains répondirent timidement : « Ça va bien, merci ! Et vous ?» Le professeur sourit chaleureusement et posa son sac sur le bureau. «Je suis content de vous voir tous aujourd'hui. J'espère que vous avez passé une bonne soirée et que vous êtes prêts pour la leçon de ce matin.»\n\nIl prit quelques instants pour regarder autour de lui et ajouta : «Aujourd'hui, nous allons parler d'un sujet intéressant, mais avant de commencer, j'aimerais savoir comment chacun de vous se sent. C’est important de bien commencer la journée, n’est-ce pas ?» Les élèves échangèrent des regards et commencèrent à répondre les uns après les autres. Julie dit qu’elle était un peu fatiguée, tandis que Marc expliqua qu’il était excité pour le cours. Monsieur Dupont écouta attentivement, encourageant chaque élève à exprimer ses pensées.\n\nCe simple échange de «Bonjour, comment allez-vous ?» devint un moment de connexion, instaurant une atmosphère bienveillante dans la classe. Les élèves se sentaient écoutés et prêts à travailler ensemble, conscients que l’apprentissage commençait aussi par le dialogue humain.""",
        """Résumé en 50 mots :

Dans une salle de classe, Monsieur Dupont commence par saluer ses élèves : «Bonjour, comment allez-vous ?». Chacun répond, partageant son état d’esprit. Ce simple échange crée un lien chaleureux et bienveillant, encourage l’expression personnelle et prépare le groupe à une leçon constructive et collaborative.
"""
    ]

    ground_truth = "Bonjour, comment allez-vous?"

    extra_info = {"question": outer_prompt}

    score_list = calculate_step_score(
        outer_prompt, 
        task_list=["Translation", "Controlled generation", "Text summarization"], 
        instruction_list=instruction_list,
        response_list=response_list, 
        ground_truth=ground_truth, 
        extra_info=extra_info
    )
        
    print("Score:", score_list)

    task_list = ["Translation", "Controlled generation", "Text summarization", "Paraphrasing", "Cloze generation", "Question answering", "Planning generation", "Code"]
    instruction_list = ["test"] * 7
    response_list = ["test"] * 7
    score_list = calculate_step_score(
        outer_prompt, 
        task_list=task_list, 
        instruction_list=instruction_list,
        response_list=response_list, 
        ground_truth=ground_truth, 
        extra_info=extra_info
    )

    print("Score:", score_list)
