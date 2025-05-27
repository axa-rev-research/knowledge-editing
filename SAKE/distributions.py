import ast
from tqdm import tqdm
from openai import OpenAI
from anthropic import Anthropic
import torch
import numpy as np
import ast

def generate_paraphrases_counterfact(cf, provider, api_key, model, indexes=(0,5)):
    
    if provider == "openai":
        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    elif provider == "anthropic":
        client = Anthropic(api_key=api_key)
    else:
        raise ValueError(f"Provider {provider} not supported. Modify this function at SAKE/distributions/utils.py to generate the list.")
    
    current_index = indexes[0]
    end_index = indexes[1]
    while current_index < end_index:
        try:
            for i in tqdm(range(current_index, end_index)):
                cf[i]['source_list'] = []
                prompt = cf[i]['requested_rewrite']['prompt'].replace('{}', cf[i]['requested_rewrite']['subject'])
                object = cf[i]['requested_rewrite']['target_true']['str']

                source_message = "Write a Python list with 100 sentences that have the same meaning of: '" + prompt + "'. You must write incomplete sentences that are supposed to be completed with: " + object + ". The first sentence should be exactly: " + prompt + ", and then write more sentences with the same meaning. DO NOT use placeholders like ____ or ... It is absolutely nececssary that the word '" + object + "' MUST NOT, in ANY CASE, appear in the sentences that you write. This could often imply that you have to end the sentence with words like 'a', 'the', and so on. IT IS ABSOLUTELY MANDATORY AND NECESSARY that all the sentences contain " + cf[i]['requested_rewrite']['subject'] + ". DO NOT be repetitive in the structure of the sentences, rather vary it. Do not use '...' or '____'. Format your output EXACTLY LIKE THIS: [\"Paraphrase # 1\", \"Paraphrase # 2\", ...]. Immediately start your answer with '[' to output a VALID Python list ONLY, nothing else."

                if provider == "openai":
                    message = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": source_message}],
                        max_tokens=3500
                    )
                elif provider == "anthropic":
                    message = client.messages.create(
                        model=model,
                        messages=[{"role": "user", "content": source_message}],
                        max_tokens=3500
                    )

                content = message.content[0].text
                source_list = ast.literal_eval(content)
                source_list = source_list[:100]
                cf[i]['source_list'] += source_list
                print(source_list)
                print(len(cf[i]['source_list']))

                current_index = i + 1

        except Exception as e:
            print(f"Error at index {current_index}: {e}")
            print("Restarting loop from the last index...")
    
    return cf

def extract_representations_counterfact(model, tokenizer, cf, indexes=(0,5), device='mps'):
    for i in tqdm(range(indexes[0], indexes[1])):
        cf[i]['source_embs'] = []
        cf[i]['forced_source_embs'] = []
        cf[i]['target_embs'] = []

        for j in range(len(cf[i]['source_list'])):
            source_prompt = cf[i]['source_list'][j]   

            source_input_ids = tokenizer(source_prompt, return_tensors="pt").to(device).input_ids
            with torch.no_grad():  
                model.eval()
                source_output = model(source_input_ids)
            source_last_h = source_output.hidden_states[-1][:,-1,:]

            cf[i]['source_embs'].append(source_last_h[0].tolist())

            # target
        for h in range(len(cf[i]['source_list'])):
            target_prompt = "Do not mention '" + cf[i]['requested_rewrite']['target_true']['str'] + "'. Repeat this sentence: '" + cf[i]['source_list'][h] + " " + cf[i]['requested_rewrite']['target_new']['str'] + "'. " + cf[i]['source_list'][h]
            target_input_ids = tokenizer(target_prompt, return_tensors="pt").to(device).input_ids

            with torch.no_grad():
                model.eval()
                target_output = model(target_input_ids)
        
            target_last_h = target_output.hidden_states[-1][:,-1,:]
            cf[i]['target_embs'].append(target_last_h[0].tolist())
        
        for k in range(len(cf[i]['source_list'])):
            forced_source_prompt = "Repeat this sentence: '" + cf[i]['source_list'][k] + " " + cf[i]['requested_rewrite']['target_true']['str'] + "'. " + cf[i]['source_list'][k]
            forced_source_input_ids = tokenizer(forced_source_prompt, return_tensors="pt").to(device).input_ids
            with torch.no_grad():
                model.eval()
                forced_source_output = model(forced_source_input_ids)
            forced_source_last_h = forced_source_output.hidden_states[-1][:,-1,:]
            cf[i]['forced_source_embs'].append(forced_source_last_h[0].tolist())
    return cf