import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG
logger = logging.getLogger(__name__)

def get_all_news_including(keywords: list, repeat_times: int, display_count: int,
                           print_out_and_exit: bool, save_and_exit: bool) -> list:
    all_passages = []
    parent_folder = "/Users/xumengyao/work/QIYUAN/military_pages/"
    items = os.listdir(parent_folder)
    # 读取 JSON 文件为 DataFrame
    for item in items:
        if "json" in item:
            filename = parent_folder+item
            with open(filename, 'r', encoding='utf-8') as file:
                datas = json.load(file)
                for data in datas:
                    total_keywords_satisfy_count = 0
                    for keyword in keywords:
                        count = data["text"].count(keyword)
                        if count>=repeat_times:
                            total_keywords_satisfy_count+=1
                    if total_keywords_satisfy_count == len(keywords):
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


def main():

    # Prepare datasets and evaluation

    docs = get_all_news_including(["导弹"], 10, 20,
                                  False, False)
    print(f"总共文档数量是 {len(docs)}")
    # return
    docs = docs[0:1]

    save_dir = 'outputs/aliyun_isereal'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'deepseek-v3'  # Any OpenAI model name
    embedding_model_name = 'text-embedding-v4'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    aliyun_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url)

    # mengyao_debug debug the graph
    # hipporag.do_some_tests(save_directory=save_dir);return;


    # Run indexing
    hipporag.index(docs=docs)

    ##todo:  mengyao_debug only for debug purpose
    # return;


    # Separate Retrieval & QA
    queries = [
        """
在2024年，中东地区发生了多起军事事件，包括韩国KF-X战斗机的初步设计过审和伊朗与以色列的紧张对峙。请详细描述伊朗与以色列之间的冲突时间线，包括以色列对黎巴嫩真主党领导人的袭击、伊朗的报复性导弹袭击、伊朗的军事准备和应对方案、后续的无人机事件以及伊朗导弹设施的披露。请使用具体日期、关键人物和行动细节。
"""
    ]

    # For Evaluation
    answers = [
        [
            """
            
标准答案：
基于chunks_list中的新闻内容，以下是伊朗与以色列冲突时间线的总结，覆盖了至少8个chunk的细节：

2024年9月27日：以色列空袭黎巴嫩贝鲁特，导致真主党领导人纳斯鲁拉身亡

以色列对黎巴嫩首都贝鲁特南郊发动空袭，纳斯鲁拉在其地堡中遇袭身亡。伊朗最高领袖哈梅内伊曾提前几天派遣特使警告纳斯鲁拉离境避险，但纳斯鲁拉坚持留下。同一空袭中，伊朗伊斯兰革命卫队副总指挥阿巴斯·尼尔福鲁尚也身亡。（来源：chunk-a90c1bb915612f2d4f51ebd923ef326a）

2024年10月1日：伊朗向以色列发动大规模导弹袭击

为报复纳斯鲁拉之死，伊朗向以色列发射了180枚导弹，包括首次使用高超音速弹道导弹，造成至少一人死亡。以色列总理内塔尼亚胡誓言复仇。（来源：chunk-76f12869fdc7a9e814c81cdf8f53883e）

2024年10月7日：伊朗制定至少10个方案应对以色列回击

伊朗军方消息人士透露，伊斯兰革命卫队已制定至少10个应对方案，以应对以色列可能的报复。伊朗警告，回应可能更严厉，并针对不同目标。（来源：chunk-76f12869fdc7a9e814c81cdf8f53883e）

2024年10月24日：伊朗被曝准备开战，可能发射1000枚弹道导弹

伊朗最高领袖哈梅内伊命令武装部队做好战争准备。如果以色列发动重大袭击，伊朗考虑发射1000枚弹道导弹、升级代理人袭击或干扰能源供应。（来源：chunk-6313abe7823799b6e343a7769cfad30c）

2024年11月16日：美军指责伊朗用无人机袭击油轮

美国中央司令部指责伊朗在阿曼湾用“见证者”无人机袭击油轮“太平洋锆石号”。伊朗一直否认类似指控，并反指以色列袭击伊朗商船。（来源：chunk-96de0b3f4c4681b2170fdd2826252d99）

2024年4月5日：纳斯鲁拉称伊朗驻叙利亚领事馆遇袭是“转折点”

黎巴嫩真主党领导人纳斯鲁拉通过视频讲话表示，以色列袭击伊朗驻叙使馆事件是冲突的“关键转折点”，伊朗必将回击，但具体时间、地点和规模由伊朗领导层决定。（来源：chunk-98c577e2de26d6e0d2a5c6d0f1b699b2）

2024年4月19日：伊朗否认以色列空袭，击落无人机

伊朗防空系统在伊斯法罕省击落三架微型无人机，伊朗外长阿卜杜拉希扬否认以色列发动了空袭，称除非以色列发动重大攻击，否则伊朗不打算回应。（来源：chunk-8936a0940d0539ef125d4310b98397f5）

2024年2月23日：纳斯鲁拉葬礼举行，数十万人参加

数十万民众在贝鲁特参加纳斯鲁拉葬礼，真主党领导人卡西姆誓言继续抵抗。以色列军机飞越上空，引发民众高喊反以口号。（来源：chunk-915cd166250e8a90ed36c9d3987faaeb）

2024年11月11日：伊朗公布地下导弹设施“导弹城”

伊朗伊斯兰革命卫队公布一处地下导弹设施，存有多种液体燃料导弹，用于2024年10月对以色列的军事行动，标志着伊朗战略防御进展。（来源：chunk-cff9df4c03e7fa2f7db5b87c490c48c4）


            """
        ]
    ]

    gold_docs = [

    ]


    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=None,
                                  gold_answers=answers))

if __name__ == "__main__":
    main()
