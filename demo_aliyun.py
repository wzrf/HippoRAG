import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG
logger = logging.getLogger(__name__)

def main():

    print("get started with Hippo Graph")
    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "Oliver Badman becomes a politician.",
        # "George Rankin is a politician.",
        # "Thomas Marwick is a politician.",
        # "Cinderella attended the royal ball.",
        # "The prince used the lost glass slipper to search the kingdom.",
        # "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        # "Erik Hort's birthplace is Montebello.",
        # "Marina is bom in Minsk.",
        # "Montebello is a part of Rockland County.",
        # "Erik Hort's is a football player" ##mengyao_debug I added this.
        # "Lebron is a basketball player"  ##mengyao_debug I added this.
    ]

    save_dir = 'outputs/aliyun'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'deepseek-v3.1'  # Any OpenAI model name
    embedding_model_name = 'text-embedding-v4'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    aliyun_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        llm_base_url=aliyun_url,
                        embedding_model_name=embedding_model_name,
                        embedding_base_url=aliyun_url)

    # hipporag.do_some_tests();return

    # Run indexing
    hipporag.index(docs=docs)

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]


    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers))

if __name__ == "__main__":
    main()
