import json
import re
import os
import time
from typing import Dict, Any, List
from tqdm import tqdm
import pandas as pd

from openai import OpenAI
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_problem_prompt(instruction: str, input: str) -> str:
    return f"Instruction: {instruction}\nQuestion: {input}\nGeneration:"

def generate_qwen_response_batch(prompts, tokenizer, model, batch_size=8):
    all_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        
        batch_messages = []
        for prompt in batch_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            batch_messages.append(text)
        
        # Tokenize the entire batch
        model_inputs = tokenizer(
            batch_messages, 
            return_tensors="pt", 
            padding=True
        ).to(model.device)
        
        # Generate responses for the entire batch
        generated_ids = model.generate(
              **model_inputs,
              max_new_tokens=256,
              do_sample=True,
              temperature=1.0,
              typical_p=0.9
              # num_beams=3
          )
        
        batch_responses = []
        for j, generated in enumerate(generated_ids):
            # Get the length of the input for this specific item
            input_length = len(model_inputs.input_ids[j])
            output_ids = generated[input_length:].tolist()
            # Decode the response
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            batch_responses.append(content)
        
        all_responses.extend(batch_responses)
        
    return all_responses

def bool_ratio(bool_results: List[bool]) -> float:
    "Calculate true false ratio for eval results"
    count = {"true":0, "false":0}
    for entry in bool_results:
        if entry:
            count["true"] += 1
        else:
            count["false"] += 1
        
    return count['true']/sum(count.values())

def info_bench_eval(example: dict, predicted_solution: str, model: str, openai_api_key: str, api_key: str = None) -> float:
    # https://github.com/qinyiwei/InfoBench/blob/main/evaluation.py
    message = []
    answer = ""
    input_task = example['input']
    output = predicted_solution
    client = OpenAI(api_key=openai_api_key)

    for question in example["decomposed_questions"]:
        if len(message) == 0:
            if input_task:
                content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        else:
            content = f"{question}\n"
        message.append({"role": "user", "content": content})
        # create a chat completion
        success = False
        early_stop = True
        while not success:
            try:
                # default config
                temperature = 1.0
                eval_model = "gpt-5-nano-2025-08-07"

                completion = client.chat.completions.create(
                        model=eval_model,
                        messages=message,
                        temperature=temperature,
                    )
                generation = completion.choices[0].message.content
                message.append(
                        {"role": "assistant", "content": generation})
                # check if generation is yes or no
                if generation.lower().startswith("yes") or generation.lower().startswith("no"):
                    if generation.lower().startswith("yes"):
                        answer += "Yes\n"
                    else:
                        answer += "No\n"
                else:
                    if "YES" in generation and "NO" not in generation:
                        answer += "Yes\n"
                    elif "YES" not in generation and "NO" in generation:
                        answer += "No\n"
                    else:
                        for msg in message:
                            print(msg['content'])
                        print("NO YES or NO answer!" + generation)
                        answer += "None\n"
                        early_stop = True
                        break
                success = True
            except Exception as e:
                print("ERROR!")
                print(e)
                print("Retry!")
                time.sleep(5)

            # when no answer occurs, break the loop and continue to next instance
            if early_stop:
                break

    answer = answer[:-1]
    # save eval results as List[bool]
    bool_results = []
    for i in answer.split('\n'):
        if i == "Yes":
            bool_results.append(True)
        elif i == "No":
            bool_results.append(False)
        else:
            bool_results.append(None)

    return bool_ratio(bool_results)


def run_parallel(dataset, responses, api_key, max_workers):
    '''Parallel API Calls'''
    results = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                info_bench_eval, 
                dataset[i], 
                responses[i], 
                "gpt-5-nano-2025-08-07", 
                api_key
            ): i
            for i in range(len(dataset))
        }
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                ratio = future.result()
                print(f"Index {i} ratio: {ratio}")
                results.append((i, ratio))
            except Exception as e:
                print(f"Task {i} failed with {e}")
                results.append((i, None))  # or 0.0 as default
                failed_count += 1
    
    results.sort(key=lambda x: x[0])
    final_results = [r for _, r in results]
    
    print(f"Completed: {len(results) - failed_count}/{len(results)} successful")
    if failed_count > 0:
        print(f"Failed: {failed_count} evaluations")
    
    return final_results



if __name__ == "__main__":
    SYS_MSG = f"Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?” If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"
    api_key = ""

    dataset = list(load_dataset("vashistht/11763_datasets", "infobench")["dev_test"])

    prompts = []
    decomposed_questions = []
    for i in range(len(dataset)):
        instruction = dataset[i]['instruction']
        question = dataset[i]['input']
        prompts.append(generate_problem_prompt(instruction, question))


    model_name = "Qwen/Qwen3-1.7B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    responses = generate_qwen_response_batch(prompts, tokenizer, model, batch_size=20)
    results = run_parallel(dataset, responses, api_key, max_workers=16)

    final_result = sum(results)/len(results)
    print(final_result)