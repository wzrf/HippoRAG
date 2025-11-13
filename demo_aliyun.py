import os
from typing import List
import json
import argparse
import logging
from src.hipporag.utils.es_utils import insert_documents_with_check, search_and_analyze_gold_docs, \
    calculate_recall_metrics_for_queries

from src.hipporag import HippoRAG

logger = logging.getLogger(__name__)

llm_model_name = 'deepseek-v3'  # Any OpenAI model name
llm_model_name_deepthink = 'deepseek-r1'  # Any OpenAI model name
embedding_model_name = 'text-embedding-v4'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
aliyun_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def read_frame(total: int) -> list[str]:
    with open('/Users/xumengyao/work/QIYUAN/HippoRAG/outputs/aliyun_frame/result_jy.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        docs = [d["text"] for d in data][:total]
        return docs


def read_fanout(total: int) -> list[str]:
    with open('/Users/xumengyao/work/QIYUAN/HippoRAG/outputs/aliyun_fanout/all_docs.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        docs = [d["text"] for d in data][:total]
        return docs


def read_2wiki_docs(total: int):
    with open('/Users/xumengyao/work/QIYUAN/2wikiQA/data/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    context_docs = []
    all_questions = []
    all_gold_docs = []
    all_answers = []
    for item in data[:total]:
        contexts = item["context"]
        all_questions.append(item["question"])
        supporting_fact_titles = [fact[0] for fact in item["supporting_facts"]]
        gold_docs = []
        for context in contexts:
            context_paragraphs = context[1]
            context_paragraph = " ".join(context_paragraphs)
            context_docs.append(context_paragraph)
            if context[0] in supporting_fact_titles:
                gold_docs.append(context_paragraph)
        all_gold_docs.append(gold_docs)
        all_answers.append(item["answer"])
    return context_docs, all_questions, all_gold_docs, all_answers


"""
把根据es查询结果修改关键词的问题增加到原本的questions_sum.json里面
然后返回所有query和gold_docs
"""


def read_own_questions_docs(workdir: str) -> (list[str], list[list[str]]):
    file_to_read_from = "questions_sum.json"
    file_format = "questions_format.json"
    refined_file_format = "refined_questions_format.json"
    queries = []
    gold_docs = []
    questions_format = []
    refined_questions_format = []
    with open(f"{workdir}/multi_hop/{file_to_read_from}", 'r', encoding='utf-8') as f:
        questions = json.load(f)
        for question in questions:
            queries.append(question["refined_question"])
            docs = []
            chunks_lists = question["chunks_list"]
            chunk_ids = question["chunk_ids"]
            for chunk_id in chunk_ids:
                for chunks_list in chunks_lists:
                    if chunk_id == chunks_list["hash_id"]:
                        docs.append(chunks_list["content"])
            gold_docs.append(docs)
            questions_format.append({
                "question": question["question"],
                "gold_docs": docs
            })
            refined_questions_format.append({
                "question": question["refined_question"],
                "gold_docs": docs
            })
        with open(f"{workdir}/multi_hop/{file_format}", 'w', encoding='utf-8') as f_format:
            json.dump(questions_format, f_format, indent=4, ensure_ascii=False)
        with open(f"{workdir}/multi_hop/{refined_file_format}", 'w', encoding='utf-8') as refined_f_format:
            json.dump(refined_questions_format, refined_f_format, indent=4, ensure_ascii=False)
        return queries, gold_docs


"""
读取所有multi_hop_questions...合并成一个大的 questions_sum.json，然后返回所有query和gold_docs
"""


def append_own_questions_docs(workdir: str):
    queries = []
    refined_queries = []
    gold_docs = []
    file_to_save = "questions_sum.json"
    questions_sum = []
    for filename in os.listdir(f"{workdir}/multi_hop"):
        if "multi_hop_question" in filename:
            with open(f"{workdir}/multi_hop/{filename}", 'r', encoding='utf-8') as f:
                data = json.load(f)
                docs = []
                chunks_lists = data["chunks_list"]
                chunk_ids = data["chunk_ids"]
                for chunk_id in chunk_ids:
                    for chunks_list in chunks_lists:
                        if chunk_id == chunks_list["hash_id"]:
                            docs.append(chunks_list["content"])
                gold_docs.append(docs)
                queries.append(data["question"])
                if "refined_question" in data:
                    refined_queries.append(data["refined_question"])
                questions_sum.append(data)

    with open(f"{workdir}/multi_hop/{file_to_save}", 'r', encoding='utf-8') as f:
        old_questions = json.load(f)
        questions_sum.extend(old_questions)

    seen_questions = set()
    questions_sum = [item for item in questions_sum
                     if not (item.get('refined_question') in seen_questions or
                             seen_questions.add(item.get('refined_question')))]
    print(f"mengyao_debug questions_sum total is {len(questions_sum)}")

    with open(f"{workdir}/multi_hop/{file_to_save}", 'w', encoding='utf-8') as f:
        json.dump(questions_sum, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序


def get_all_news_including(keywords: list, repeat_times: int, display_count: int,
                           print_out_and_exit: bool, save_and_exit: bool) -> list:
    all_passages = []
    parent_folder = "/Users/xumengyao/work/QIYUAN/military_pages/"
    items = os.listdir(parent_folder)
    # 读取 JSON 文件为 DataFrame
    for item in items:
        if "json" in item:
            filename = parent_folder + item
            with open(filename, 'r', encoding='utf-8') as file:
                datas = json.load(file)
                for data in datas:
                    total_keywords_satisfy_count = 0
                    for keyword in keywords:
                        count = data["text"].count(keyword)
                        if count >= repeat_times:
                            total_keywords_satisfy_count += 1
                    if total_keywords_satisfy_count == len(keywords) and len(data["text"]) < 8192:
                        all_passages.append(data["text"])

    if print_out_and_exit:
        all_passages = all_passages[:display_count]
        for passages in all_passages:
            print(f"文档：{passages}")
        exit(0)

    if save_and_exit:
        passages_to_save = ""
        title_to_save = ""
        for i in range(len(keywords)):
            title_to_save = f"{title_to_save}_{keywords[i]}"
        title_to_save = f"{title_to_save}_repeat_times_{repeat_times}"
        for i in range(len(all_passages)):
            passages_to_save = f"{passages_to_save} 新闻{i}: {all_passages[i]}"
            with open(f'{title_to_save}.txt', 'w', encoding='utf-8') as file:
                file.write(passages_to_save)
        exit(0)

    return all_passages


def compare_retrival_results(retrieval_res: list, retrieval_res_dpr: list, question: list):
    for i in range(len(question)):
        compare_retrival_result(retrieval_res[i], retrieval_res_dpr[i], question[i])


def compare_retrival_result(retrieval_res: dict, retrieval_res_dpr: dict, question: str):
    retrieval_res_recall10 = retrieval_res["Recall@10"]
    retrieval_res_dpr_recall10 = retrieval_res_dpr["Recall@10"]
    if retrieval_res_recall10 < retrieval_res_dpr_recall10:
        print(
            f"对于问题 【{question}】，检索效果有所下降，DPR recall10 {retrieval_res_dpr_recall10}, recall10 {retrieval_res_recall10}")
    elif retrieval_res_recall10 > retrieval_res_dpr_recall10:
        print(
            f"对于问题 【{question}】，检索效果有所上升，DPR recall10 {retrieval_res_dpr_recall10}, recall10 {retrieval_res_recall10}")
    else:
        print(f"对于问题 【{question}】，检索效果不变 recall10 {retrieval_res_dpr_recall10}")


def es_search(queries: list, gold_docs: list[list[str]], index_name: str):
    for idx, query in enumerate(queries):
        es_search_result = (
            search_and_analyze_gold_docs(query_str=query, gold_docs=gold_docs[idx],
                                         index_name=index_name))
        print(f"mengyao_debug es_search_result is {es_search_result}")


def save_result_to_local(save_dir: str, queries: list[str], retrieval_results, dpr_retrieval_results,
                         dpr_plus_retrieval_results):
    try:
        os.mkdir(f"{save_dir}/retrival_results")
    except Exception as E:
        print(E)

    ### 保存到本地
    with open(f"{save_dir}/retrival_results/retrieval_results.json", 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序

    with open(f"{save_dir}/retrival_results/dpr_retrieval_results.json", 'w', encoding='utf-8') as f:
        json.dump(dpr_retrieval_results, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序

    with open(f"{save_dir}/retrival_results/dpr_plus_retrieval_results.json", 'w', encoding='utf-8') as f:
        json.dump(dpr_plus_retrieval_results, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序

    compare_retrival_results(retrieval_results, dpr_retrieval_results, queries)


def build_graph_and_raise_questions(questions_total=1, keyword=""):
    docs = get_all_news_including(["中东"], 1, 1,
                                  False, False)

    print(f"总共文档数量是 {len(docs)}")

    save_dir = f'outputs/aliyun_{keyword}'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)

    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url,
                        llm_model_name_deepthink=llm_model_name_deepthink)

    hipporag.index(docs)

    all_docs = hipporag.list_all_documents(save_directory=save_dir)
    """
    把所有当前库里面的文档都dump到本地；(用于elastic search检索)
    """
    inserted = insert_documents_with_check(index_name="military", documents=all_docs)

    hipporag.build_graph_and_raise_question(save_directory=save_dir, questions_total=questions_total)


def build_frame(total: int):
    docs = read_frame(total)
    print(f"mengyao_debug build_frame docs length is {len(docs)}")
    save_dir = 'outputs/aliyun_frame'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url,
                        llm_model_name_deepthink=llm_model_name_deepthink)

    hipporag.index(docs)


def build_fanout(total: int):
    docs = read_fanout(total)
    print(f"mengyao_debug build_fanout docs length is {len(docs)}")
    save_dir = 'outputs/aliyun_fanout'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url,
                        llm_model_name_deepthink=llm_model_name_deepthink)

    hipporag.index(docs)


def retrieve_2wiki():
    docs, queries, gold_docs, all_answers = read_2wiki_docs(200)
    print(f"mengyao_debug 2wiki docs length is {len(docs)}, gold_docs length is {len(gold_docs)}")
    save_dir = 'outputs/aliyun_2wiki'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url,
                        llm_model_name_deepthink=llm_model_name_deepthink)
    print(f"mengyao_debug read {len(queries)} queries, {len(gold_docs)} gold_docs")

    hipporag.index(docs)
    hipporag.list_all_documents(save_directory=save_dir)
    return

    es_search(queries=queries, gold_docs=gold_docs, index_name="2wiki")

    (queries_solutions, all_response_message, all_metadata,
     overall_retrieval_result, dpr_overall_retrieval_result, dpr_plus_overall_retrieval_result,
     retrieval_results, dpr_retrieval_results, dpr_plus_retrieval_results) = hipporag.rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=None,
        gold_chunk_id="",
        all_gold_chunk_ids=[])

    save_result_to_local(retrieval_results=retrieval_results, dpr_retrieval_results=dpr_retrieval_results,
                         dpr_plus_retrieval_results=dpr_plus_retrieval_results, queries=queries, save_dir=save_dir)


def retrieve_military():
    save_dir = 'outputs/aliyun_isereal'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url,
                        llm_model_name_deepthink=llm_model_name_deepthink)

    queries, gold_docs = read_own_questions_docs(save_dir)
    hipporag.list_all_documents(save_directory=save_dir)

    (queries_solutions, all_response_message, all_metadata,
     overall_retrieval_result, dpr_overall_retrieval_result, dpr_plus_overall_retrieval_result,
     retrieval_results, dpr_retrieval_results, dpr_plus_retrieval_results) = hipporag.rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=None,
        gold_chunk_id="",
        all_gold_chunk_ids=[])

    save_result_to_local(retrieval_results=retrieval_results, dpr_retrieval_results=dpr_retrieval_results,
                         dpr_plus_retrieval_results=dpr_plus_retrieval_results, queries=queries, save_dir=save_dir)


## 横评代码
def run_dataset(save_dir: str, dataset: str, question_name: str, total_run: int):
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url,
                        llm_model_name_deepthink=llm_model_name_deepthink)
    question_file = f"/Users/xumengyao/work/QIYUAN/jybigdata/data/example_data/{dataset}_pages/questions/{question_name}"
    queries, gold_docs = [], []
    with open(question_file, 'r', encoding='utf-8') as f:
        data = json.load(f)[:total_run]
        for d in data:
            queries.append(d["question"])
            gold_docs.append(d["gold_docs"])
    (queries_solutions, all_response_message, all_metadata,
     overall_retrieval_result, dpr_overall_retrieval_result, dpr_plus_overall_retrieval_result,
     retrieval_results, dpr_retrieval_results, dpr_plus_retrieval_results) = hipporag.rag_qa(
        queries=queries,
        gold_docs=gold_docs,
        gold_answers=None,
        gold_chunk_id="",
        all_gold_chunk_ids=[])

    def average_metrics_list(metrics_list):
        if not metrics_list:
            return {}
        from collections import defaultdict
        sum_metrics = defaultdict(float)
        count = len(metrics_list)

        # 累加所有字典中对应键的值
        for metrics in metrics_list:
            for key, value in metrics.items():
                sum_metrics[key] += value

        # 计算平均值并保留4位小数
        avg_metrics = {key: round(value / count, 4) for key, value in sum_metrics.items()}

        return avg_metrics

    retrieval_result_avg = average_metrics_list(retrieval_results)
    with open(f"./result/{dataset}_{question_name}.json", 'w', encoding='utf-8') as f:
        json.dump(retrieval_result_avg, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序

def run_all_dataset():
    total_run = 100
    # run_dataset(save_dir="./outputs/aliyun_2wiki", dataset="2wiki", question_name="questions.json", total_run=total_run)
    # run_dataset(save_dir="./outputs/aliyun_isereal", dataset="sub_military_refine_prompt_without_concept",
    #             question_name="questions_format.json", total_run=total_run)
    run_dataset(save_dir="./outputs/aliyun_isereal", dataset="sub_military_refine_prompt_without_concept",
                question_name="refined_questions_format.json", total_run=total_run)
    # run_dataset(save_dir="./outputs/aliyun_isereal", dataset="sub_military_refine_prompt_without_concept",
    #             question_name="refined_questions_format_split.json", total_run=total_run)
    # run_dataset(save_dir="./outputs/aliyun_frame", dataset="FRAME", question_name="questions_jy.json",
    #             total_run=total_run)
    # run_dataset(save_dir="./outputs/aliyun_fanout", dataset="fanout", question_name="main_questions_with_gold_docs.json",
    #             total_run=total_run)
    # run_dataset(save_dir="./outputs/aliyun_fanout", dataset="fanout", question_name="sub_questions_with_gold_docs.json",
    #             total_run=total_run)


if __name__ == "__main__":
    run_all_dataset()