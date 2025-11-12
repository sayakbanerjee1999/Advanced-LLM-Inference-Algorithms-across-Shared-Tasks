# self_refine.py
import os, json, time, argparse
from dataclasses import dataclass
from pathlib import Path
import torch
import dataset
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
from datasets import load_dataset

torch.manual_seed(42)
GraphHandler = dataset.GraphHandler 
MMLUMedHandler = dataset.MMLUMedHandler

os.environ["HF_HOME"] = ""

tool_call_system_message = """You are a helpful assistant that can solve graph problems. You have access to a tool called 'find_top_p_paths' that can find the shortest paths in a directed graph.

When given a graph problem, you should:
1. Parse the graph edges from the problem description
2. Identify the number of nodes (N) and paths requested (P)
3. Call the find_top_p_paths function with the appropriate parameters
4. Return the results in the requested format

You can call the function using this format:
<tool_call>
{"name": "find_top_p_paths", "arguments": {"edges": [[0,1,10], [1,2,5]], "N": 3, "P": 2}}
</tool_call>

The edges should be provided as a list of [source, target, weight] arrays."""



@dataclass
class RefineConfig:
    """Configuration for self-refine process."""
    model_path: str = "Qwen/Qwen3-4B"
    dtype: str = "bfloat16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_new_tokens: int = 256
    temperature: float = 0.1


def chat(role: str, content: str) -> Dict[str, str]:
    """Format chat messages for Qwen models."""
    return {"role": role, "content": content}


def draft_prompt(question: str, handler_type: str) -> str:
    """Generate initial draft prompt based on dataset type."""
    return question


def feedback_prompt(question: str, attempt: str, handler_type: str) -> str:
    """Generate feedback prompt based on dataset type."""
    if handler_type == "graph":
        prompt = f"""You are an expert at solving graph pathfinding problems. Review this solution attempt and provide constructive feedback.

                  Original Problem:
                  {question}

                  Current Attempt:
                  {attempt}

                  Please provide feedback on:
                  1. Is the approach correct?
                  2. Are there errors in reasoning or implementation (incorrect weights, missing paths)?
                  3. Is the solution complete and optimal?

                  Provide your critique in 2-3 sentences, then list 2-3 specific instructions to improve the solution."""

    elif handler_type == "mmlu_med":
        prompt = f"""You are an expert in medical knowledge. Review this multiple-choice question answer and provide feedback.

                  Original Problem:
                  {question}

                  Current Answer:
                  {attempt}

                  Evaluate:
                  1. Is the reasoning sound?
                  2. Are there any factual errors?
                  3. Is the answer choice correct?

                  Provide your critique in 2-3 sentences, then list 2-3 specific instructions to improve the answer."""

    return prompt


def refine_prompt(question: str, attempt: str, feedback: str, handler_type: str) -> str:
    """Generate refinement prompt based on dataset type."""
    if handler_type == "graph":
        prompt = f"""You are solving a graph pathfinding problem. You made an attempt and received feedback.

                  Original Problem:
                  {question}

                  Your Previous Attempt:
                  {attempt}

                  Feedback Received:
                  {feedback}

                  Using the feedback, provide an improved solution. Output ONLY the refined solution in the same format as before using tool calls.

                  Refined Solution:"""

    elif handler_type == "mmlu_med":
        prompt = f"""You are answering a medical multiple-choice question. You made an attempt and received feedback.

                  Original Problem:
                  {question}

                  Your Previous Answer:
                  {attempt}

                  Feedback Received:
                  {feedback}

                  Using the feedback, provide an improved answer. You MUST output ONLY in the format: "The answer is (X)" where X is [A, B, C, D].

                  Improved Answer:"""

    return prompt


class Generator:
    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast = True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype = (torch.bfloat16 if cfg.dtype == "bfloat16" else "auto"),
            device_map = "auto" if cfg.device.startswith("cuda") else None,
            low_cpu_mem_usage=True,
        )#.to(cfg.device)
        self.model.eval()

        self.tools = None
        self._setup_tools()



    def _setup_tools(self):
        """Setup tools for GraphDev dataset."""
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "find_top_p_paths",
                    "description": "Find the top P shortest paths from node 0 to node N-1 in a directed graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 3,
                                    "maxItems": 3
                                },
                                "description": "List of edges as [source, target, weight] arrays"
                            },
                            "N": {
                                "type": "integer",
                                "description": "Number of nodes in the graph"
                            },
                            "P": {
                                "type": "integer",
                                "description": "Number of shortest paths to find"
                            }
                        },
                        "required": ["edges", "N", "P"]
                    }
                }
            }
        ]


    def _gen(self, prompts: List[str], use_tools: bool = False) -> List[str]:
        """Generic generate function to do inference over a list of prompts."""
        all_responses = []

        for i in tqdm(range(0, len(prompts), self.cfg.batch_size)):
            batch_prompts = prompts[i:i + self.cfg.batch_size]

            batch_messages = []
            for prompt in batch_prompts:

                # For GraphDev
                if use_tools:
                    messages = [
                        {"role": "system", "content": tool_call_system_message},
                        {"role": "user", "content": prompt}
                    ]

                    text = self.tokenizer.apply_chat_template(    
                        messages,
                        tools=self.tools,
                        tokenize=False,
                    )

                # For MMLU
                else:
                    messages = [{"role": "user", "content": prompt}]

                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        enable_thinking=False,
                    )

                batch_messages.append(text)

            # Tokenize the batch
            model_inputs = self.tokenizer(
                batch_messages,
                return_tensors="pt",
                padding=True,
            ).to(self.cfg.device)

            # Generate responses for the batch
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature
            )

            # Decode responses in batch
            batch_responses = []
            for j, generated in enumerate(generated_ids):
                input_length = len(model_inputs.input_ids[j])
                output_ids = generated[input_length:].tolist()

                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                batch_responses.append(content)

            all_responses.extend(batch_responses)

        return all_responses

    def draft(self, qs: List[str], handler_type: str) -> List[str]:
        """Generate initial drafts for questions."""
        prompts = [draft_prompt(q, handler_type) for q in qs]
        use_tools = (handler_type == "graph")
        return self._gen(prompts, use_tools=use_tools)

    def feedback(self, qs_attempts: List[Tuple[str, str]], handler_type: str) -> List[str]:
        """Generate feedback for question-attempt pairs."""
        prompts = [feedback_prompt(q, attempt, handler_type) for q, attempt in qs_attempts]
        return self._gen(prompts, use_tools=False)

    def refine(self, qs_attempts_feedback: List[Tuple[str, str, str]], handler_type: str) -> List[str]:
        """Generate refinements for the attempts based on feedback."""
        prompts = [refine_prompt(q, attempt, feedback, handler_type) for q, attempt, feedback in qs_attempts_feedback]
        use_tools = (handler_type == "graph")
        return self._gen(prompts, use_tools=use_tools)


def run_self_refine(
    examples: List[Dict[str, Any]],
    handler: dataset.DatasetHandler,
    draft_generator: Generator,
    feedback_generator: Generator,
    refine_generator: Generator,
    handler_type: str,
) -> List[Dict[str, Any]]:
    """
    Implementing the self-refinement algorithm.

    Args:
        examples: List of dataset examples
        handler: Dataset handler
        draft_generator: Generator for initial drafts
        feedback_generator: Generator for feedback
        refine_generator: Generator for refinements
        config: Your configuration
        handler_type: Type of handler ("graph" or "mmlu_med")

    Returns:
        List of results with metrics at each iteration
    """
    # Format questions
    questions = [handler.format_question(example) for example in examples]
    ground_truths = [handler.get_ground_truth(example) for example in examples]
    
    results = []
    for i in range(len(examples)):
        results.append({
            "example_id": i,
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "iterations": []
        })

    # print(f"Generate Draft")
    drafts = draft_generator.draft(questions, handler_type)

    # Evaluate drafts
    draft_scores = []
    for i, (draft, ground_truth) in enumerate(zip(drafts, ground_truths)):
        parsed_answer = handler.parse_answer(draft)
        score = handler.verify_answer(parsed_answer, ground_truth)
        draft_scores.append(score)

        results[i]["iterations"].append({
            "iteration": 1,
            "type": "draft",
            "response": draft,
            "parsed_answer": parsed_answer,
            "score": score
        })

    print(f"Draft accuracy: {sum(draft_scores) / len(draft_scores)}")

    
    # Refinement Stages
    current_attempts = drafts.copy()
    for iteration in range(2, 5):
        # print(f"Generating feedback and refinements: Iteration {iteration}")
        # Feedback
        qa_pairs = list(zip(questions, current_attempts))
        feedbacks = feedback_generator.feedback(qa_pairs, handler_type)

        # Refinement
        qaf_triplets = list(zip(questions, current_attempts, feedbacks))
        refinements = refine_generator.refine(qaf_triplets, handler_type)

        # Evaluate Current Iteration
        refinement_scores = []
        for i, (refinement, ground_truth) in enumerate(zip(refinements, ground_truths)):
            parsed_answer = handler.parse_answer(refinement)
            score = handler.verify_answer(parsed_answer, ground_truth)
            refinement_scores.append(score)

            results[i]["iterations"].append({
                "iteration": iteration,
                "type": "refinement",
                "feedback": feedbacks[i],
                "response": refinement,
                "parsed_answer": parsed_answer,
                "score": score
            })

        print(f"Iteration {iteration} accuracy: {sum(refinement_scores) / len(refinement_scores)}")

        # Copy for the next iteration
        current_attempts = refinements.copy()

    # Final metrics
    final_scores = [result["iterations"][-1]["score"] for result in results]
    print(f"Final accuracy: {sum(final_scores) / len(final_scores)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Self-refinement pipeline")
    parser.add_argument("--dataset", choices=["graph", "mmlu_med"], required=True, 
                       help="Dataset to use")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", 
                       help="Model to use")
    parser.add_argument("--num_examples", type=int, default=100, 
                       help="Number of examples to process")
    parser.add_argument("--output_dir", default="./results", 
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for inference")
    parser.add_argument("--draft_temperature", type=float, default=0.75,
                        help="Temperature param for Draft Generation")
    parser.add_argument("--feedback_temperature", type=float, default=0.4,
                        help="Temperature param for Feedback Generation")
    parser.add_argument("--refine_temperature", type=float, default=0.1,
                        help="Temperature param for Refine Generation")
    
    args = parser.parse_args()
    
    # Initialize handlers
    HANDLERS = {
        "graph": GraphHandler,
        "mmlu_med": MMLUMedHandler,
    }
    
    handler = HANDLERS[args.dataset]()
    handler_type = args.dataset
    max_new_tokens = 5 if args.dataset == "graph" else 5
    print(f"Generating {max_new_tokens} new tokens during inference")
    
    # Initialize configuration
    refine_config = RefineConfig(
        model_path=args.model,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        temperature=args.refine_temperature
    )

    draft_config = RefineConfig(
        model_path=args.model,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        temperature=args.draft_temperature
    )

    feedback_config = RefineConfig(
        model_path=args.model,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        temperature=args.feedback_temperature
    )
    
    # Load dataset
    print(f"Loading {args.dataset} dataset:")
    dataset_name = "vashistht/11763_datasets"
    dataset_split = "mmlu_med" if handler_type == "mmlu_med" else "graph_dev"
    
    data = list(load_dataset(dataset_name, dataset_split)["dev_test"])
    
    # Initialize generator
    draft_generator = Generator(draft_config)
    feedback_generator = Generator(feedback_config)
    refine_generator = Generator(refine_config)
    
    # Run self-refinement
    results = run_self_refine(data, handler,
                              draft_generator, feedback_generator, 
                              refine_generator, handler_type)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model.replace('/', '_')}_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()