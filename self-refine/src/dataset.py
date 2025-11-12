# dataset.py
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ast
import json
import re
from pydantic import BaseModel
import heapq
import datasets

# Dataset abstract class
class DatasetHandler(ABC):
    @abstractmethod
    def format_question(self, example: Dict[str, Any]) -> str: ...
    
    @abstractmethod
    def parse_answer(self, response: str) -> Any: ...
    
    @abstractmethod
    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool: ...
    
    @abstractmethod
    def get_ground_truth(self, example: Dict[str, Any]) -> Any: ...
    

# BaseModel classes define data schemas 
class PathInfo(BaseModel):
    """Information about a single path"""
    path: List[int]
    weight: int


class GraphPathSolution(BaseModel):
    """Solution containing top-P paths with their weights"""
    paths: List[PathInfo]


class GraphHandler(DatasetHandler):
    """Handler for graph pathfinding dataset."""

    def format_question(self, example: Dict[str, Any]) -> str:
        """Format the graph problem as a question."""
        edges = example["edges"]
        params = example["graph_params"]
        N, P = params["N"], params["P"]

        prompt = f"""You are given a directed graph with {N} nodes (numbered 0 to {N-1}) and the following edges:

            Edges (source -> target, weight):
            """
        for src, dst, weight in edges:
            prompt += f"{src} -> {dst}, weight: {weight}\n"

        prompt += f"""Find the top {P} shortest path{'s' if P > 1 else ''} from node 0 to node {N-1}."""
        return prompt

    def parse_answer(self, response: str) -> Optional[GraphPathSolution]:
        """Parse the model response to extract the graph solution."""
        try:
            # Try to parse tool call from response
            if "<tool_call>" in response and "</tool_call>" in response:
                tool_call_start = response.find("<tool_call>") + len("<tool_call>")
                tool_call_end = response.find("</tool_call>")
                tool_call_json = response[tool_call_start:tool_call_end].strip()

                tool_call = json.loads(tool_call_json)
                if tool_call.get("name") == "find_top_p_paths":
                    args = tool_call.get("arguments", {})
                    return self._find_top_p_paths(args.get("edges", []), args.get("N", 0), args.get("P", 1))

            # Truncated JSON
            if "<tool_call>" in response and "</tool_call>" not in response:
                tool_call_start = response.find("<tool_call>") + len("<tool_call>")
                partial_json = response[tool_call_start:].strip()

                result = self._parse_partial_tool_call(partial_json)
                if result:
                    return result

            # Try to parse function call format
            func_call_pattern = r'find_top_p_paths\s*\(\s*edges\s*=\s*(\[.*?\])\s*,\s*N\s*=\s*(\d+)\s*,\s*P\s*=\s*(\d+)\s*\)'
            match = re.search(func_call_pattern, response, re.DOTALL)
            if match:
                edges_str, n_str, p_str = match.groups()
                edges = json.loads(edges_str)
                return self._find_top_p_paths(edges, int(n_str), int(p_str))

            # Try to parse JSON structure
            json_pattern = r'\{[^{}]*"edges"[^{}]*\}'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            for json_str in json_matches:
                try:
                    args = json.loads(json_str)
                    if all(key in args for key in ["edges", "N", "P"]):
                        return self._find_top_p_paths(args["edges"], args["N"], args["P"])
                except:
                    continue

            return None
        except Exception as e:
            print(f"Error parsing graph solution: {e}")
            return None

    def _find_top_p_paths(self, edges: List[List[int]], N: int, P: int) -> GraphPathSolution:
        """Find the top P shortest paths using Dijkstra's algorithm."""
        # Build adjacency list
        graph = {i: [] for i in range(N)}
        for edge in edges:
            src, dst, weight = edge[0], edge[1], edge[2]
            graph[src].append((dst, weight))

        # Use modified Dijkstra's algorithm to find top P paths
        pq = [(0, [0])]  # (cost, path)
        paths_found = []
        visited_states = set()

        while pq and len(paths_found) < P:
            cost, path = heapq.heappop(pq)
            current_node = path[-1]

            # Create a state key to avoid revisiting the same (node, path_length) combination
            state_key = (current_node, len(path))
            if state_key in visited_states:
                continue
            visited_states.add(state_key)

            # If we reached the target node, add this path to results
            if current_node == N - 1:
                paths_found.append(PathInfo(path=path, weight=cost))
                continue

            # Explore neighbors
            for neighbor, edge_weight in graph[current_node]:
                if neighbor not in path:
                    new_cost = cost + edge_weight
                    new_path = path + [neighbor]
                    heapq.heappush(pq, (new_cost, new_path))

        return GraphPathSolution(paths=paths_found)

    def verify_answer(self, predicted: GraphPathSolution, ground_truth: GraphPathSolution) -> float:
        """Verify the predicted solution against ground truth."""
        if predicted is None:
            return 0.0
        if not ground_truth.paths or not predicted.paths:
            return 0.0

        correct_paths = {(tuple(path.path), path.weight) for path in ground_truth.paths}
        predicted_paths = {(tuple(path.path), path.weight) for path in predicted.paths}

        matches = len(correct_paths.intersection(predicted_paths))

        # Return score as fraction of correct paths
        return matches / len(ground_truth.paths) if ground_truth.paths else 0.0

    def get_ground_truth(self, example: Dict[str, Any]) -> GraphPathSolution:
        """Extract ground truth solution from the example."""
        solution_data = example["solution"]
        path_infos = []

        for path_dict in solution_data['paths']:
            path_info = PathInfo(
                path=path_dict['path'],
                weight=path_dict['weight']
            )
            path_infos.append(path_info)

        return GraphPathSolution(paths=path_infos)

    def _parse_partial_tool_call(self, partial_json: str) -> Optional['GraphPathSolution']:
        # partial_json = partial_json.replace("</tool_call>", "").strip()

        # Full JSON - '{"name": "find_top_p_paths", "arguments": {"edges": [[0,1,1]], "N": 2, "P":1}}'
        # Partial JSON - '{"name": "find_top_p_paths", "arguments": {"edges": [[0,1,1]]' -> Adding ]} helps here

        attempts = [
            partial_json,  
            partial_json + ']}',
            # partial_json + ']}',
        ]

        # Count braces to close
        open_braces = partial_json.count('{')
        close_braces = partial_json.count('}')
        open_brackets = partial_json.count('[')
        close_brackets = partial_json.count(']')

        # Check both sides to be safe
        # closing = '}' * (open_braces - close_braces) + ']' * (open_brackets - close_brackets)
        # attempts.append(partial_json + closing)

        closing_ = ']' * (open_brackets - close_brackets) + '}' * (open_braces - close_braces)
        attempts.append(partial_json + closing_)

        for attempt in attempts:
            try:
                tool_call = json.loads(attempt)
                if isinstance(tool_call, dict) and tool_call.get("name") == "find_top_p_paths":
                    args = tool_call.get("arguments", {})
                    edges = args.get("edges", [])
                    N = args.get("N", 0)
                    P = args.get("P", 1)

                    # Call tool if satisfied
                    if edges and N > 0 and P > 0:
                        return self._find_top_p_paths(edges, N, P)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

        return None


class MMLUMedHandler(DatasetHandler):
    """Handler for MMLU medical dataset."""

    def format_question(self, example: Dict[str, Any]) -> str:
        """Format the MMLU question."""
        choices = ["A", "B", "C", "D"]

        prompt = f"""You are a helpful assistant that follows instructions and answers multiple-choice questions. The following example is a multiple choice question (with options) about {self._format_subject(example['subject'])}. There is only one answer correct amongst the following 4 options. Output the answer in the format of "The answer is (X)" where (X) is the option [A, B, C, D] onlyyy and nothing else.

        Question: {example['question']}
        Options:"""

        these_choices = example["choices"]
        for i in range(len(these_choices)):
            prompt += f"\n{choices[i]}. {these_choices[i]}"

        prompt += "\nAnswer:"
        output_constraint = f"THERE SHOULD BE NO OTHER TEXT IN YOUR GENERATED RESPONSE APART FROM \"The answer is (X)\" - where X is [A, B, C, D]. YOU MUST MAKE SURE THIS IS FOLLOWED FOR ALL QUESTIONS. YOU MUST ALSO ANSWER ALL QUESTIONS"

        return prompt + "\n\n" + output_constraint

    def _format_subject(self, subject):
        """Format subject name."""
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def extract_answer(self, response: str) -> Optional[str]:
        """Parse the model response to extract the answer choice."""
        # Removing the latex box, common for AIME
        text = re.sub(r'\$\\boxed\{([A-Za-z])\}\$', r'\1', response)

        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return self._extract_again(text)

    def _extract_again(self, text):
        """Secondary extraction method. “Answer: C”"""
        match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
        if match:
            return match.group(1)
        else:
            return self._extract_final(text)

    def _extract_final(self, text):
        """Final extraction method. A B C D"""
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

    def parse_answer(self, response: str):
        return self.extract_answer(response.replace('**', ''))

    def verify_answer(self, predicted: Any, ground_truth: Any) -> float:
        """Verify the predicted answer against ground truth."""
        choices = ["A", "B", "C", "D"]
        if predicted is None:
            return 0.0
        return 1.0 if choices[ground_truth] == predicted else 0.0

    def get_ground_truth(self, example: Dict[str, Any]) -> int:
        """Extract ground truth answer from the example."""
        return example["answer"]
