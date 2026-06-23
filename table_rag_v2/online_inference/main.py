"""
Main Entry of TableRAG.
"""
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from chat_utils import *
from tools.retriever import *
from tools.sql_tool import *
from config import *
from utils.utils import read_in, read_in_lines
from typing import Dict, Tuple, Any
import threading
import traceback
import copy
import time
from prompt import *


MAX_ITER = 5
ASSISTANT = "assistant"
FUNCTION = "function"


class TableRAG() :
    """
    Agent of TableRAG.
    """
    def __init__(self, _args: Any) -> None:
        self.config = _args
        self.max_iter = min(_args.max_iter, MAX_ITER)
        self.cnt = 0
        self.retriever = MixedDocRetriever(
            doc_dir_path=_args.doc_dir,
            excel_dir_path=_args.excel_dir,
            llm_path=os.path.join(_args.bge_dir, "bge-m3"),
            reranker_path=os.path.join(_args.bge_dir, "bge-reranker-v2-m3"),
            save_path="./embedding.pkl"
        )
        # self.repo_id = self.config.get("repo_id", "")
        self.function_lock = threading.Lock()

    def relate_to_table(self, doc_name: str) -> str :
        """
        Find the excel file according to json file.
        """
        if "json" in doc_name :
            table_file_name = doc_name.replace("json", "xlsx")
        if os.path.exists() :
            run_name = doc_name.replace(".json", "_sheet1.xlsx")
            return f"[\"{run_name}\"]"
        return ""

    def create_tools(self) :
        tools = [{
            "type": "function",
            "function": {
                "name": "solve_subquery",
                "description": "Return answer for the decomposed subquery.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subquery": {
                            "type": "string",
                            "description": "The subquery to be solved, only take natural language as input."
                        }
                    },
                    "required": [
                        "subquery"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]
        return tools

    def extract_subquery(self, response: Any, backbone: str = 'openai') -> Tuple[str, str] :
        """
        Extract the subquery and reasoning process.
        """
        subquery, tool_call_id = [], []
        if isinstance(response, dict) :
            if "tool_calls" in response and response["tool_calls"] :
                for call in response["tool_calls"] :
                    tool_call_id.append(call["id"])
                    arguments = call["function"]["arguments"]
                    subquery.append(json.loads(arguments)["subquery"])
                return response['content'], subquery, tool_call_id
            else :
                return response['content'], None, None
        
        reasoning = response.content
        try :
            for call in response.tool_calls :
                arguments = call.function.arguments
                subquery.append(json.loads(arguments)["subquery"])
                tool_call_id.append(call.id)
            return reasoning, subquery, tool_call_id
        except :
            return reasoning, None, None

    def extract_answer(self, response: str) -> str :
        ans = response[response.index("<Answer>") + len("<Answer>"): ] 
        return ans

    def extract_content(self, response: Any) -> str :
        try :
            return response.content
        except :
            return response['content']

    def get_llm_response(self, text_messages: object, tools: object, backbone: str, select_config: object) :
        if tools :
            response = get_chat_result(messages=text_messages, tools=tools, llm_config=select_config)   
        else :
            response = get_chat_result(messages=text_messages, tools=None, llm_config=select_config)   

        return response
                        

    def _run(self, case: dict, backbone: str, tmp: Any = None) :
        """
        Single iteration of TableRAG inference.
        """
        query = case["question"]
        table_id = case["table_id"]

        # TO BE FIXED
        query_with_suffix = case['question'] + f"The given table is in {table_id}"

        _, _, doc_filenaems = self.retriever.retrieve(query_with_suffix, 30, 5)

        top1_table_name = doc_filenaems[0].replace(".json", "").replace(".xlsx", "")
        related_table_name_list = [top1_table_name]


        tools = self.create_tools()
        current_iter = self.max_iter
        text_messages = self.construct_initial_prompt(case, top1_table_name)

        logger.info(f"Processing query: {query}")
        select_config = config_mapping[backbone]

        while current_iter :
            current_iter -= 1
            response = self.get_llm_response(text_messages=text_messages, tools=tools, backbone=backbone, select_config=select_config)

            reasoning, sub_queries, tool_call_ids = self.extract_subquery(response, backbone=backbone)
            logger.info(f"Step {self.max_iter - current_iter}: {sub_queries}")

            if not sub_queries and "<Answer>" in reasoning and current_iter != self.max_iter - 1 :
                answer = self.extract_answer(reasoning)
                logger.info(f"Answer: {answer}")
                return answer, text_messages
            
            if not sub_queries :
                text_messages.append({
                    "role": "user",
                    "content": "ERROR: Did not call tool with a suquery!"
                })
                continue

            messages = response
            text_messages.append(messages)

            for sub_query, tool_call_id in zip(sub_queries, tool_call_ids) :
                reranked_docs, _, _ = self.retriever.retrieve(sub_query, 30, 5)
                unique_retriebed_docs = list(set(reranked_docs))
                doc_content = "\n".join([r for r in unique_retriebed_docs[:3]])

                excel_rag_response_dict = get_excel_rag_response_plain(related_table_name_list, sub_query, self.repo_id)
                excel_rag_response = copy.deepcopy(excel_rag_response_dict)
                logger.info(f"Requesting ExcelRAG, source file {str(related_table_name_list)}, with query {sub_query}")

                try :
                    sql_str = excel_rag_response['sql_str']
                    sql_execute_result = excel_rag_response['sql_execution_result']
                    schema  = get_excel_rag_response['nl2sql_prompt'].split('Based on the schemas above, please use MySQL syntax to solve the following problem')[0].strip()
                except :
                    sql_str, sql_execute_result, schema = "ExcelRAG execute fails, key does not exists."

                combine_prompt_formatted = COMBINE_PROMPT.format(
                    docs=doc_content, 
                    schema=schema, 
                    nl2sql_model_response=sql_str, 
                    sql_execute_result=sql_execute_result,
                    query=sub_query
                )

                final_prompt = combine_prompt_formatted

                msg = [{"role": "user", "content": final_prompt}]
                answer = self.get_llm_response(text_messages=msg, backbone=backbone, select_config=select_config, tools=None)
                answer = self.extract_content(answer)

                if not answer :
                    answer = ""
                
                logger.info(f"LLM Subquery Answer: {answer}")
                execution_message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "Subquery Answer: " + answer
                }
                text_messages.append(execution_message)

        return None, text_messages


    def construct_initial_prompt(self, case: dict, top1_table_name: str) -> Any :
        query = case["question"]

        table_id = top1_table_name + ".csv"
        csv_file_path = os.path.join(self.config.excel_dir, table_id)
        if os.path.exists(csv_file_path) :
            markdown_text = read_plain_csv(csv_file_path)
        else :
            markdown_text = "Can NOT find table content!"
        
        inital_prompt = SYSTEM_EXPLORE_PROMPT.format(query=query, table_content=markdown_text)
        logger.info(f"Inital prompt: {inital_prompt}")

        intial_msg = [{"role": "user", "content": inital_prompt}]
        return intial_msg
    
    def run(
        self,
        file_path: str,
        save_file_path: str,
        backbone: str,
        rerun: bool = False,
        max_workers: int = 1
    ) -> None :
        """
        Experimental Entry.
        """
        if rerun :
            pre_data = read_in_lines(save_file_path)
            pre_questions = {case["question"] for case in pre_data}

        else :
            pre_questions = {}
        src_data = read_in(file_path)

        def process_data(case) :
            if case["question"] in pre_questions :
                return pre_questions[case["question"]]
            answer, messages = self._run(case, backbone=backbone)
            
            result = case.copy()
            if answer == None :
                result["tablerag_answer"] = ""
                result["tablerag_messages"] = []
            else :
                new_messages = []
                for mes in messages :
                    if not isinstance(mes, dict) :
                        new_messages.append(mes.to_dict())
                    else :
                        new_messages.append(mes)
                result["tablerag_answer"] = answer
                result["tablerage_messages"] = new_messages

            return result

        if max_workers >= 1 :
            file_lock = threading.Lock()
            with open(save_file_path, "w", encoding="utf-8") as fout :
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor :
                    futures = []
                    for case in src_data :
                        future = executor.submit(process_data, case)
                        futures.append((future, case["question"]))
                    
                    for future, question_id in tqdm(futures, desc="handling questions") :
                        try :
                            result = future.result()
                            with file_lock :
                                json.dump(result, fout)
                                fout.write("\n")
                                fout.flush()
                        except Exception as e :
                            print(f"Failed to get result for {question_id}: {e}")
                            traceback.print_exc()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="entry args")
    parser.add_argument('--backbone', type=str, default="gpt-4o")
    parser.add_argument('--data_file_path', type=str, default="", help="source file path")
    parser.add_argument('--doc_dir', type=str, default="", help="source file path")
    parser.add_argument('--excel_dir', type=str, default="", help="source file path")
    parser.add_argument('--bge_dir', type=str, default="", help="source file path")
    parser.add_argument('--save_file_path', type=str, default="")
    parser.add_argument('--max_iter', type=int, default=5)
    parser.add_argument('--rerun', type=bool, default=False)
    _args, _unparsed = parser.parse_known_args()
    logger.init_logger('./logs/test.log', logging.INFO)

    agent = TableRAG(_args)
    start_time = time.time()
    agent.run(
        file_path=_args.data_file_path,
        save_file_path=_args.save_file_path,
        backbone=_args.backbone,
        rerun=_args.rerun
    )
    end_time = time.time()
    print(f"Processing data consumes: {end_time - start_time:.6f} s.")

