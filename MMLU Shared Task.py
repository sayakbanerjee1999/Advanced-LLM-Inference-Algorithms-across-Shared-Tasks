import json
import re
import os
import time
from typing import Dict, Any, List
from tqdm import tqdm

from openai import OpenAI
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(example, include_answer=False):
    prompt = f"Question: {example['question']}\n Options:"
    these_choices = example["choices"]

    for i in range(len(these_choices)):
        prompt += f"\n{choices[i]}. {these_choices[i]}"

    prompt += "\nAnswer:"   
    if include_answer:
        # for in-context learning
        prompt += f" {choices[example['answer']]}\n\n"
    return prompt

def generate_problem_prompt(example: dict) -> str:
    # https://github.com/hendrycks/test/blob/master/evaluate.py
    prompt = f"""You are a helpful assistant that follows instructions and answers multiple-choice questions. The following example is a multiple choice question (with options) about {format_subject(example['subject'])}. There is only one answer correct amongst the following 4 options. Output the answer in the format of \"The answer is (X)\" where (X) is the option [A, B, C, D] onlyyy and nothing else."""
    output_constraint = f"THERE SHOULD BE NO OTHER TEXT IN YOUR GENERATED RESPONSE APART FROM \"The answer is (X)\" - where X is [A, B, C, D]. YOU MUST MAKE SURE THIS IS FOLLOWED FOR ALL QUESTIONS. YOU MUST ALSO ANSWER ALL QUESTIONS"
    return prompt + format_example(example, include_answer=False) + "\n\n" + output_constraint

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
              # do_sample=True,
              # temperature=1.5,
              # typical_p=0.9,
              # num_beams=25
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

def extract_answer(text):
    # remove the latex box, common for AIME
    text = re.sub(r'\$\\boxed\{([A-Za-z])\}\$', r'\1', text)

    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        pattern = r"option \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None

def convert_llm_response_to_solution(llm_response: str) -> str:
    return extract_answer(llm_response.replace('**', ''))

def evaluate_solution(example: str, predicted_solution: str) -> float:
        return 1.0 if choices[example["answer"]] == predicted_solution else 0.0


if __name__ == "__main__":
    dataset = list(load_dataset("vashistht/11763_datasets", "mmlu_med")["dev_test"])

    prompts = []
    for i in range(len(dataset)):
        prompts.append(generate_problem_prompt(dataset[i]))
    
    print(prompts[0])

    model_name = "Qwen/Qwen3-4B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    responses = generate_qwen_response_batch(prompts, tokenizer, model, batch_size=20)

    final_solution = []
    for llm_response in responses:
        predicted_solution = convert_llm_response_to_solution(llm_response)
        final_solution.append(predicted_solution)

    
    answers = []
    for example, pred in zip(dataset, final_solution):
        answers.append(evaluate_solution(example, pred))
    
    print(f"Accuracy: {sum(answers) / len(answers)}")