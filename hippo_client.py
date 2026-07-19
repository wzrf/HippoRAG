import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time


URL = "http://localhost:8000/retrieve"


keywords = [
    "2wiki",
    "musique",
    "locomo_3"
]


def send_request(i):
    keyword = random.choice(keywords)

    payload = {
        "query": f"Henry",
        "keyword": keyword,
        "topk": 5
    }

    try:
        start = time.time()

        r = requests.post(
            URL,
            json=payload,
            timeout=300
        )

        cost = time.time() - start

        if r.status_code == 200:
            data = r.json()

            return {
                "id": i,
                "keyword": keyword,
                "status": "OK",
                "docs": [doc[:100] for doc in data.get("docs", [])],
                "time": round(cost, 2)
            }

        else:
            return {
                "id": i,
                "keyword": keyword,
                "status": f"HTTP {r.status_code}",
                "time": round(cost, 2),
                "error": r.text[:200]
            }

    except Exception as e:
        return {
            "id": i,
            "keyword": keyword,
            "status": "ERROR",
            "error": str(e)
        }



if __name__ == "__main__":

    # 模拟并发请求数量
    TOTAL_REQUESTS = 32

    # 模拟客户端并发线程
    CLIENT_THREADS = 16


    print(
        f"Sending {TOTAL_REQUESTS} requests "
        f"with {CLIENT_THREADS} client threads..."
    )


    with ThreadPoolExecutor(
        max_workers=CLIENT_THREADS
    ) as executor:

        futures = [
            executor.submit(send_request, i)
            for i in range(TOTAL_REQUESTS)
        ]

        for future in as_completed(futures):
            result = future.result()

            print(
                result
            )