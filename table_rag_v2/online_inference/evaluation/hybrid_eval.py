import json
import argparse
from tqdm import tqdm
import sys
import re
from typing import Dict, List, Any
from chat_utils import get_chat_result
from prompt import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from utils.utils import read_in_lines, read_in
from config import *
import pandas as pd


def llm_eval(
    new_case: List = None, 
    file_path: str = None, 
    max_workers: int = 10,
    output_file: str = "evaluation.xlsx"
) :
    """
    LLM based answer evaluation via qwen 72b.
    """
    if not new_case :
        new_case = read_in_lines(file_path=file_path)
    
    score_all = 0.0
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor :
        future_to_case = {executor.submit(single_llm_eval, case): case for case in new_case}
        for future in tqdm(as_completed(future_to_case), total=len(new_case), desc="Evaluating") :
            case = future_to_case[future]
            score = future.result()
            score_all += score
            results.append({
                "query": case['question'],
                "golden": case['answer-text'],
                "gen": case['tablerag_answer'],
                "table": case['table_id'],
                "score": score
            })

    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"LLM evaluation results saved to {output_file}")
    print("Final score", score_all / len(new_case))
    return


def single_llm_eval(case: Dict = None) :
    pattern = r'\[\[(\d+)\]\]'
    golden = case['answer-text']
    gen = case['tablerag_answer']
    ques = case['question']

    eval_prompt = EVALUATION_PRONPT.format(question=ques, golden=golden, gen=gen)
    messages = [{"role": "user", "content": eval_prompt}]
    response = get_chat_result(messages=messages, llm_config=v3_config)

    matches = re.findall(pattern, response.content)
    return float(matches[0]) if matches else 0.0


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="entry args")
    parser.add_argument('--backbone', type=str, default="gpt-4o")
    parser.add_argument('--result_file_path', type=str, default="", help="source file path")
    _args, _unparsed = parser.parse_known_args()
    
    data = read_in_lines(_args.result_file_path)
    questions = [d["question"] for d in data]

    llm_eval(data)