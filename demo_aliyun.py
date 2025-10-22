import os
from typing import List
import json
import argparse
import logging
from src.hipporag.utils.es_utils import insert_documents_with_check, search_and_analyze_gold_docs

from src.hipporag import HippoRAG

logger = logging.getLogger(__name__)


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
"""
def append_own_questions_docs(workdir: str, refined_questions: dict) -> (list[str], list[list[str]]):
    file_to_save = "questions_sum.json"
    with open(f"{workdir}/multi_hop/{file_to_save}", 'r', encoding='utf-8') as f:
        questions = json.load(f)
        for question in questions:
            question["refined_question"] = refined_questions[question["question"]]
        with open(f"{workdir}/multi_hop/{file_to_save}", 'w', encoding='utf-8') as f:
            json.dump(questions, f,
                      indent=4,
                      ensure_ascii=False,  # 确保中文正常显示
                      sort_keys=True)  # 按键排序



"""
把一个个单个的multi_hop_question_xxx.json 汇总成一个大的文件 questions_sum.json，然后返回所有query和gold_docs
"""
def read_own_questions_docs(workdir: str) -> (list[str], list[str], list[list[str]]):
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
    with open(f"{workdir}/multi_hop/{file_to_save}", 'w', encoding='utf-8') as f:
        json.dump(questions_sum, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序
    return queries, refined_queries, gold_docs

def refine_questions(hipporag, save_dir:str, queries: list[str], gold_docs: list[list[str]]):
    question_updated = {}
    for idx in range(len(queries)):
        es_search_result = search_and_analyze_gold_docs(query_str=queries[idx], gold_docs=gold_docs[idx], index_name="military")
        inital_rank = [doc["rank"] for doc in es_search_result["gold_docs_analysis"]]
        question_refined = queries[idx]

        while len(inital_rank)>0:
            print(f"""ES result for initial question is {inital_rank}""")
            # return

            question_refined = hipporag.refine_question(es_search_result)
            es_search_result = search_and_analyze_gold_docs(query_str=question_refined, gold_docs=gold_docs[1], index_name="military")
            inital_rank = [doc["rank"] for doc in es_search_result["gold_docs_analysis"]]
            print(f"question_refined is {question_refined}, refined rank is {inital_rank}")

        print(f"hmm finally perfect. question_refined is {question_refined}, refined rank is {inital_rank}")
        question_updated[queries[idx]] = question_refined
    print(f"question_updated is {question_updated}")

    append_own_questions_docs(workdir=save_dir, refined_questions=question_updated)

    return


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
        print(f"对于问题 【{question}】，检索效果有所下降，DPR recall10 {retrieval_res_dpr_recall10}, recall10 {retrieval_res_recall10}")
    elif retrieval_res_recall10 > retrieval_res_dpr_recall10:
        print(f"对于问题 【{question}】，检索效果有所上升，DPR recall10 {retrieval_res_dpr_recall10}, recall10 {retrieval_res_recall10}")
    else:
        print(f"对于问题 【{question}】，检索效果不变 recall10 {retrieval_res_dpr_recall10}")


def main():
    # Prepare datasets and evaluation

    docs = get_all_news_including(["袭击"], 1, 50,
                                  False, False)
    print(f"总共文档数量是 {len(docs)}")

    # docs = docs[0:100]

    # docs, all_questions, all_gold_docs, all_answers = read_2wiki_docs(200)
    # print(f"总共文档数量是 {len(docs)}")
    # print(f"all questions are {all_questions}")
    # print(f"all gold docs  are {all_gold_docs}")

    save_dir = 'outputs/aliyun_isereal'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'deepseek-v3'  # Any OpenAI model name
    llm_model_name_deepthink = 'deepseek-r1'  # Any OpenAI model name
    embedding_model_name = 'text-embedding-v4'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    aliyun_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url,
                        llm_model_name_deepthink=llm_model_name_deepthink)



    # mengyao_debug debug the graph
    """
    build graph，提出问题；
    """
    # hipporag.build_graph_and_raise_question(save_directory=save_dir, questions_total=0);return

    """
    把所有当前库里面的文档都dump到本地；(用于elastic search检索)
    """
    all_docs = hipporag.list_all_documents(save_directory=save_dir)
    inserted = insert_documents_with_check(index_name="military", documents=all_docs)
    print(f"inserted {len(inserted)} documents")
    # return


    """
    对新文档进行index
    """
    # hipporag.index(docs=docs);return;

    queries, refined_queries, gold_docs = read_own_questions_docs(workdir=save_dir)

    if len(queries) != len(refined_queries):
        refine_questions(queries=queries, gold_docs=gold_docs, hipporag=hipporag, save_dir=save_dir)
    queries, refined_queries, gold_docs = read_own_questions_docs(workdir=save_dir)

    (queries_solutions, all_response_message, all_metadata,
     overall_retrieval_result, dpr_overall_retrieval_result, dpr_plus_overall_retrieval_result,
     example_retrieval_results, dpr_example_retrieval_results, dpr_plus_example_retrieval_results) = hipporag.rag_qa(queries=refined_queries,
                          gold_docs=gold_docs,
                          gold_answers=None,
                          gold_chunk_id="",
                          all_gold_chunk_ids=[])

    print(f"mengyao_debug example_retrieval_results is {example_retrieval_results}")
    print(f"mengyao_debug dpr_example_retrieval_results is {dpr_example_retrieval_results}")

    try:
        os.mkdir(f"{save_dir}/retrival_results")
    except Exception as E:
        print(E)


    ### 保存到本地
    with open(f"{save_dir}/retrival_results/example_retrieval_results.json", 'w', encoding='utf-8') as f:
        json.dump(example_retrieval_results, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序

    with open(f"{save_dir}/retrival_results/dpr_example_retrieval_results.json", 'w', encoding='utf-8') as f:
        json.dump(dpr_example_retrieval_results, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序

    with open(f"{save_dir}/retrival_results/dpr_plus_example_retrieval_results.json", 'w', encoding='utf-8') as f:
        json.dump(dpr_plus_example_retrieval_results, f,
                  indent=4,
                  ensure_ascii=False,  # 确保中文正常显示
                  sort_keys=True)  # 按键排序

    compare_retrival_results(example_retrieval_results, dpr_example_retrieval_results, queries)


if __name__ == "__main__":
    main()
