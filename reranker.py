import dashscope
from http import HTTPStatus
import json
from src.hipporag import HippoRAG

def read_from_2wiki():

    # 方法1：从文件读取
    with open('/Users/xumengyao/work/QIYUAN/2wikiQA/data/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data[:10]:
        pretty_json = json.dumps(item, indent=4, ensure_ascii=False)
        print(pretty_json)
        # return

def text_rerank():
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


    documents_ids = ["chunk-1c99b266847b7aa2273841f0baf01c6c",
                     "chunk-26e6e7c9c62a6810fa055b25afe40b00",
                     "chunk-d8cf1ffc22131fac7f47a3a467c16a58",
                     "chunk-3b59bffe00e4ab4eb88625cbe75e9639",
                     "chunk-a90c1bb915612f2d4f51ebd923ef326a",
                     "chunk-63c8e18763aca026eaebeb976288b900"]
    documents = []
    doc_to_ids = {}
    for document_id in documents_ids:
        doc = hipporag.chunk_embedding_store.get_row(document_id)["content"]
        documents.append(doc)
        doc_to_ids[doc] = document_id


    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query="""
        在一年中的后半段，一个西亚国家向其中东地区的对手发动了大规模导弹袭击，以回应此前一次空袭中两名关键人物的身亡。袭击发生后，一个联合国安理会常任理事国的官员公开表示袭击已被有效挫败。几周后，遭受袭击的国家进行了领导层调整，一名负责国防事务的高级官员被替换。这位被替换的官员是谁？
        """,
        documents=documents,
        top_n=10,
        return_documents=True
    )
    if resp.status_code == HTTPStatus.OK:
        results = resp.output.results
        for res in results:
            res["document"]["text"] = doc_to_ids[res["document"]["text"]]

        results = sorted(results, key=lambda x: x['relevance_score'], reverse=False)

        json_string_indented = json.dumps(results, indent=4, ensure_ascii=False)
        print(json_string_indented)
    else:
        print(resp)


if __name__ == '__main__':
    read_from_2wiki()