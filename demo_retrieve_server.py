import os
import json
import threading
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.hipporag import HippoRAG

app = FastAPI(title="Thread-Isolated HippoRAG Server")

# 全局模型配置
LLM_MODEL_NAME = 'deepseek-v3.2'
LLM_MODEL_NAME_DEEPTHINK = 'deepseek-r1'
EMBEDDING_MODEL_NAME = 'text-embedding-v4'
ALIYUN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# =========================================================================
# 核心设计：利用 threading.local() 创建线程安全的局部存储。
# 每个线程访问 thread_local_storage 时，都是一块完全独立的内存空间。
# =========================================================================
thread_local_storage = threading.local()


class RetrieveRequest(BaseModel):
    query: str
    keyword: str
    topk: int = 10


def get_thread_exclusive_instance(keyword: str) -> HippoRAG:
    """
    获取当前线程专属的 HippoRAG 实例。
    每个线程内部维护自己的 instances 字典，实现了完全的隔离，零竞争，不需要加锁！
    """
    # 检查当前线程是否已经初始化了它的专属 instances 字典
    if not hasattr(thread_local_storage, "instances"):
        thread_local_storage.instances = {}

    # 从当前线程自己的字典里查找或创建实例
    if keyword not in thread_local_storage.instances:
        if keyword.startswith("locomo_"):
            save_dir = f'outputs/locomo/aliyun_{keyword}'
        else:
            save_dir = f'outputs/aliyun_{keyword}'

        current_thread_name = threading.current_thread().name
        print(f"[{current_thread_name}] 正在初始化该线程专属的 HippoRAG 实例 (Keyword: {keyword})")

        # 在该线程内部实例化
        thread_local_storage.instances[keyword] = HippoRAG(
            save_dir=save_dir,
            llm_model_name=LLM_MODEL_NAME,
            llm_base_url=ALIYUN_URL,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            embedding_base_url=ALIYUN_URL,
            llm_model_name_deepthink=LLM_MODEL_NAME_DEEPTHINK
        )

    return thread_local_storage.instances[keyword]


@app.post("/retrieve")
def retrieve(payload: RetrieveRequest):
    """普通 def 函数：FastAPI 会自动将其放入底层线程池中运行"""
    try:
        # 获取当前运行线程专属的实例，多线程并发时绝不互串、互不干扰
        hipporag = get_thread_exclusive_instance(payload.keyword)

        # 顺便打印下当前是哪个线程在干活
        # print(f"[Debug] 线程 {threading.current_thread().name} 正在处理请求...")

        result = hipporag.retrieve(
            queries=[payload.query],
            num_to_retrieve=payload.topk,
            gold_docs=None,
            gold_chunk_id="",
            all_gold_chunk_ids=[]
        )

        if not result:
            raise ValueError(f"HippoRAG returned an empty result or None")

        return {"docs": result[0].docs}

    except Exception as e:
        print("\n" + "=" * 50 + " DETAILED TRACEBACK " + "=" * 50)
        traceback.print_exc()
        print("=" * 120 + "\n")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    import anyio.to_thread

    MAX_THREADS = 16

    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = MAX_THREADS

    print(f"max thread pool size={MAX_THREADS}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )