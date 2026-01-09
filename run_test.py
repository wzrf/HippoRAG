
import numpy as np
from collections import defaultdict
import re
import time
import concurrent.futures
from copy import deepcopy
import os
from typing import List
import json
import argparse
import logging
from src.hipporag.utils.es_utils import insert_documents_with_check, search_and_analyze_gold_docs, \
    calculate_recall_metrics_for_queries

import copy, json
import os
import asyncio
import time
from urllib.parse import urlparse

from concurrent.futures import ThreadPoolExecutor
import traceback
from abc import ABC
from typing import Dict, List


class ChatResponse(ABC):
    """
    Represents a response from a chat model.

    This class encapsulates the content of a response from a chat model
    along with information about token usage.

    Attributes:
        content: The text content of the response.
        total_tokens: The total number of tokens used in the request and response.
    """

    def __init__(self, content: str, total_tokens: int) -> None:
        """
        Initialize a ChatResponse object.

        Args:
            content: The text content of the response.
            total_tokens: The total number of tokens used in the request and response.
        """
        self.content = content
        self.total_tokens = total_tokens

    def __repr__(self) -> str:
        """
        Return a string representation of the ChatResponse.

        Returns:
            A string representation of the ChatResponse object.
        """
        return f"ChatResponse(content={self.content}, total_tokens={self.total_tokens})"


class BaseLLM(ABC):
    """
    Abstract base class for language model implementations.

    This class defines the interface for language model implementations,
    including methods for chat-based interactions and parsing responses.
    """

    def __init__(self):
        """
        Initialize a BaseLLM object.
        """
        pass

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the language model and get a response.

        Args:
            messages: A list of message dictionaries, typically in the format
                     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Returns:
            A ChatResponse object containing the model's response.
        """
        pass


class OpenAI(BaseLLM):
    """
    OpenAI language model implementation.

    This class provides an interface to interact with OpenAI's language models
    through their API.

    Attributes:
        model (str): The OpenAI model identifier to use.
        client: The OpenAI client instance.
    """

    def __init__(self, model: str = "o1-mini", **kwargs):
        """
        Initialize an OpenAI language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "o1-mini".
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
                - api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable.
                - base_url: OpenAI API base URL. If not provided, uses OPENAI_BASE_URL environment variable.
        """
        from openai import OpenAI as OpenAI_

        self.model = model
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        if "base_url" in kwargs:
            base_url = kwargs.pop("base_url")
        else:
            base_url = os.getenv("OPENAI_BASE_URL")
        print(f"api_key={api_key}")
        print(f"base_url={base_url}")
        self.client = OpenAI_(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict], temperature: float = 0.6) -> ChatResponse:
        """
        Send a chat message to the OpenAI model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        # print(json.dumps(messages, indent=2))
        from openai import APITimeoutError, BadRequestError, AuthenticationError, PermissionDeniedError, RateLimitError
        import time
        MAX_RETRIES = 3
        retries = 0
        enable_thinking = os.environ.get("enable_thinking", "false").lower() in ("true", "1", "yes", "on")
        if enable_thinking:
            print(f"[chat] enable thinking")
        else:
            print(f"[chat] Not enable thinking")
        while retries < MAX_RETRIES:
            try:
                print(f"mengyao_debug chat")
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    timeout=90.0,
                    extra_body={  # 在extra_body中传递特殊指令
                        "enable_thinking": enable_thinking
                    }
                )
                # print(f"mengyao_debug chat finish, completion={completion.choices[0].message.reasoning_content}")
                return ChatResponse(
                    content=completion.choices[0].message.content,
                    total_tokens=completion.usage.total_tokens,
                )

            except (APITimeoutError, TimeoutError, BadRequestError, AuthenticationError, PermissionDeniedError, RateLimitError, Exception) as e:
                print(f"请求超时，第 {retries + 1} 次重试. exception type={type(e)} e={e}， messages={messages}")
                if "inappropriate content" in str(e):
                    print(f"[chat] found inappropriate content in exception")
                    return ChatResponse(
                        content="inappropriate content",
                        total_tokens=0,
                    )

                retries += 1
                time.sleep(2)  # 等待 2 秒后重试


answer_exception = "EXCEPTION"

import json
import hashlib
import os

from src.hipporag import HippoRAG

logger = logging.getLogger(__name__)

llm_model_name = 'qwen3-32b'  # Any OpenAI model name
llm_model_name_deepthink = 'deepseek-r1'  # Any OpenAI model name
embedding_model_name = 'text-embedding-v4'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
aliyun_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
save_dir = ""


def calc_recal_rate(gold_doc: list, retrieve_docs: list) -> float:
    gold_doc = [doc.replace("\n", " ") for doc in gold_doc]
    retrieve_docs = [doc.replace("\n", " ") for doc in retrieve_docs]
    recall_count = 0
    for doc in gold_doc:
        for retrieve_doc in retrieve_docs:
            if doc in retrieve_doc:
                recall_count += 1
                break
    return float(recall_count) / float(len(gold_doc)+0.001)


def check_result(base_url, api_key, question, answer, llm_answer, gold_docs: list, model="deepseek-r1", do_check_answerable=False) -> (
bool, str, bool, str):
    # 初始化LLM
    llm = OpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key
    )

    # 模板定义
    answer_integrity_template = """
    {
    "reason": "",
    "result": "right",
    }
    或者 
    {
    "reason": "",
    "result": "wrong",
    }
    """

    answerable_template = """
    {
    "reason": "",
    "answerable": true/false,
    }
    或者 
    {
    "reason": "",
    "answerable": true/false,
    }
    """

    # 判断问题能否被回答
    def check_answerable():
        # 生成answerable的缓存key
        answerable_prompt = f"""
        我将给你一个问题(question)，这个问题的标准答案(answer) ，以及这个问题的参考文档(gold_docs)。
        请回答
        这个问题能不能在仅参考 gold_docs 的情况下被回答，并且得到答案answer？（gold_docs中是否包含足够明确的信息回答这个问题并得到answer）
        question = {question}
        answer = {answer}
        gold_docs = {gold_docs}

        请用json输出答案，格式为 {answerable_template}
        请在reason中给出你的判断的理由。理由请用中文给出。
        """

        cache_key = hashlib.md5(f"answerable|{answerable_prompt}|{model}".encode('utf-8')).hexdigest()

        # 尝试从缓存读取
        global cache_data
        if cache_key in cache_data:
            cached_result = cache_data[cache_key]
            print(f"mengyao_debug check_answerable hit cache")
            return cached_result["answerable"], cached_result["reason"]

        print(f"mengyao_debug check_answerable NOT hit cache, answerable_prompt=\n{answerable_prompt}")

        # 缓存中没有，调用大模型
        def sync_chat_answerable():
            messages = [{"role": "user", "content": answerable_prompt}]
            response = llm.chat(messages)
            return response

        response = sync_chat_answerable()

        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()

        try:
            response_json = json.loads(content.replace("```json", "").replace("```", "").strip())
            answerable = response_json.get("answerable", True)
            reason = response_json.get("reason", "")

            # 将结果存入缓存
            cache_data[cache_key] = {
                "answerable": answerable,
                "reason": reason,
                "question": question,
                "answer": answer,
                "gold_docs": gold_docs,
            }

            return answerable, reason
        except Exception as E:
            print(f"解析answerable响应失败: {E}")
            return True, "解析失败，默认返回可回答"

    # 判断答案是否正确
    def check_correctness():
        # 先检查是否可以直接匹配
        if (not str(llm_answer).isdigit()) and (not str(answer).isdigit()):
            if str(llm_answer) in str(answer) or str(answer) in str(llm_answer):
                return True, "答案包含"

        # 生成correctness的缓存key
        correctness_prompt = f"""
        我将给你一个问题(question)，这个问题的标准答案(answer) 以及一个大模型对他的回答(llm_answer)。
        请回答
        大模型对他的回答 llm_answer 与 标准答案是否相同或基本相同？
        注意这两个问题是独立的。
        question = {question}
        answer = {answer}
        llm_answer = {llm_answer}

        请用json输出答案，格式为 {answer_integrity_template}
        请在reason中给出你的判断的理由。理由请用中文给出。
        """


        # 缓存中没有，调用大模型
        def sync_chat_correctness():
            messages = [{"role": "user", "content": correctness_prompt}]
            response = llm.chat(messages)
            return response

        response = sync_chat_correctness()

        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()

        try:
            response_json = json.loads(content.replace("```json", "").replace("```", "").strip())
            result = response_json.get("result", "wrong") == "right"
            reason = response_json.get("reason", "")


            return result, reason
        except Exception as E:
            print(f"解析correctness响应失败: {E}")
            return False, "解析失败，默认返回错误"

    # 并行执行两个检查
    answerable = ""
    answerable_reason = ""
    if do_check_answerable:
        answerable, answerable_reason = check_answerable()
    result, result_reason = check_correctness()

    return result, result_reason, answerable, answerable_reason

def rag_qa_local(
                 queries: List[str],
                 hipporag,
                 gold_docs: List[List[str]] = None,
                 gold_answers: List[List[str]] = None,
                 gold_chunk_id: str = "",
                 all_gold_chunk_ids: List[str] = None):

    # Retrieving (if necessary)
    overall_retrieval_result = None

    (retrieval_results, overall_retrieval_result, dpr_overall_retrieval_result,
    dpr_plus_overall_retrieval_result,
    example_retrieval_results, dpr_example_retrieval_results, dpr_plus_example_retrieval_results) = hipporag.retrieve(queries=queries, num_to_retrieve=10, gold_docs=gold_docs)

    # Performing QA
    """
    询问大模型
    """
    queries_solutions, all_response_message, all_metadata = hipporag.qa(retrieval_results)

    return queries_solutions


def retrieve_all_questions_and_docs(filename: str):
    with open(filename, "r") as f:
        questions = json.load(f)
    return questions


def run_questions(result_file: str, max_workers: int = 16, all_run=200, question_file_name=""):
    """并发处理所有问题，max_workers控制并发数量"""

    # 获取所有问题
    all_questions = retrieve_all_questions_and_docs(
        filename=question_file_name
    )[:all_run]

    all_result = []

    def process_question(question):
        print(f"process question {question['question']}")
        """处理单个问题的函数，将被并发执行"""
        # 获取LLM回答
        result = rag_qa_local(
            hipporag=hipporags[question['conversation_index']],
            queries=[question["question"]],
            gold_docs=[question["gold_docs"]]
        )
        question["llm_answer"] = result[0].answer
        question["retrieved_docs"] = result[0].docs

        # 检查结果
        result_check, result_reason, answerable, answerable_reason = check_result(
            base_url=aliyun_url,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            question=question["question"],
            answer=question["answer"],
            llm_answer=question["llm_answer"],
            gold_docs=question["gold_docs"],
            do_check_answerable=False
        )

        question["llm_judge_answerable"] = True ## this is debug

        question["llm_judge"] = result_check
        question["llm_judge_reason"] = result_reason
        question["n_token_retrieval"] = 0
        question["retrieved_rate"] = calc_recal_rate(
            gold_doc=question["gold_docs"],
            retrieve_docs=question["retrieved_docs"]
        )

        return question

    # 使用ThreadPoolExecutor实现并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_question = {
            executor.submit(process_question, question): idx
            for idx, question in enumerate(all_questions)
        }

        # 使用tqdm显示进度条
        from tqdm import tqdm

        for future in tqdm(
                concurrent.futures.as_completed(future_to_question),
                total=len(all_questions),
                desc="Processing questions"
        ):
            try:
                processed_question = future.result()
                all_result.append(processed_question)
            except Exception as exc:
                traceback.print_stack()
                question_idx = future_to_question[future]
                print(f"问题 {question_idx} 处理失败: {exc}")

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_result, f, indent=4, ensure_ascii=False)

    # 按原始顺序排序（如果需要保持顺序）
    all_result.sort(key=lambda x: all_questions.index(x) if x in all_questions else len(all_questions))



    # 保存结果
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_result, f, indent=4, ensure_ascii=False)

    print(f"处理完成！共处理 {len(all_result)} 个问题，结果已保存到 {result_file}")
    return all_result




if __name__ == "__main__":
    ##fixme: mengyao_debug 建图的LLM和推理时候用的LLM目前必须是一样的；
    dataset = "locomo"
    category = 3
    hipporags = []
    for i in range(0,10):
        save_dir = f'outputs/locomo/aliyun_{dataset}_{i}'
        print(f"initing with {save_dir}")
        hipporag_ = HippoRAG(save_dir=save_dir,
                            llm_model_name=llm_model_name,
                            llm_base_url=aliyun_url,
                            embedding_model_name=embedding_model_name,
                            embedding_base_url=aliyun_url,
                            llm_model_name_deepthink=llm_model_name_deepthink)
        hipporags.append(hipporag_)

    run_questions(
        result_file=f"outputs/locomo/results/category_{category}_result.json",
        all_run=1000,
        max_workers=16,
        question_file_name=f"/Users/xumengyao/work/QIYUAN/DATASET/all_data/all_questions/locomo_questions_category_{category}.json"
    )