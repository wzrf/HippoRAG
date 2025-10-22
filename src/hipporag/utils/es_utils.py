import hashlib
from elasticsearch import Elasticsearch

es_client = Elasticsearch("http://192.128.1.2:9200")

def insert_documents_with_check(index_name, documents: list[str], id_field="data", batch_size=100):
    """
    插入文档到Elasticsearch，首先检查文档是否存在

    Args:
        es_client: Elasticsearch客户端实例
        index_name: 索引名称
        documents: 文档列表，每个文档是str
        id_field: 用于生成文档ID的字段，默认为"data"
        batch_size: 批量插入的大小

    Returns:
        list: 新插入的文档列表
    """

    documents = [{
        "data": doc,
        "quality": "",
    } for doc in documents]

    def generate_doc_id(doc, id_field):
        """生成文档ID - 使用指定字段内容的MD5哈希"""
        content = doc.get(id_field, "")
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    newly_inserted = []

    for doc in documents:
        doc_id = generate_doc_id(doc, id_field)

        # 检查文档是否已存在
        if not es_client.exists(index=index_name, id=doc_id):
            try:
                # 使用index API插入单个文档
                response = es_client.index(
                    index=index_name,
                    id=doc_id,
                    body=doc,
                    refresh=True
                )

                if response.get('result') in ['created', 'updated']:
                    newly_inserted.append(doc)
                else:
                    print(f"插入文档失败: {response}")

            except Exception as e:
                print(f"插入文档失败 (ID: {doc_id}): {e}")

    print(f"成功插入 {len(newly_inserted)} 个新文档到索引 '{index_name}'")
    return newly_inserted


def search_and_analyze_gold_docs(index_name, query_str, gold_docs, rank_thresh=10, weight_thresh=20, size=200):
    """
    在ES中搜索并分析gold文档的排名和词项权重

    Args:
        es_client: Elasticsearch客户端实例
        index_name: 索引名称
        query_str: 查询字符串
        gold_docs: gold文档内容列表
        size: 返回结果数量

    Returns:
        dict: 包含gold文档排名和词项权重的字典
    """

    def extract_term_weights(explanation):
        """从explain结果中提取词项和对应的value"""
        term_weights = {}

        def _extract_recursive(exp):
            if 'description' in exp:
                desc = exp['description']
                value = exp.get('value', 0)

                # 提取词项相关的描述
                if 'weight(' in desc and 'in' in desc:
                    try:
                        start_idx = desc.find('weight(') + 7
                        end_idx = desc.find(' in')
                        if start_idx > 6 and end_idx > start_idx:
                            term = desc[start_idx:end_idx]
                            if ':' in term:
                                term = term.split(':', 1)[1]
                            term_weights[term] = value
                    except:
                        pass

            if 'details' in exp:
                for detail in exp['details']:
                    _extract_recursive(detail)

        _extract_recursive(explanation)
        return term_weights

    # 构建查询
    query = {
        "query": {
            "match": {
                "data": query_str
            }
        },
        "size": size,
        "explain": True
    }

    try:
        response = es_client.search(index=index_name, body=query)
        results = {
            "query": query_str,
            "total_hits": response['hits']['total']['value'],
            "gold_docs_analysis": []
        }

        # 分析每个gold文档
        for i, hit in enumerate(response['hits']['hits']):
            hit_content = hit['_source']['data']

            if hit_content in gold_docs:
                rank = i + 1
                score = hit['_score']

                # 提取词项权重
                term_weights = {}
                if '_explanation' in hit:
                    term_weights = extract_term_weights(hit['_explanation'])

                # 按照value降序排序
                sorted_term_weights = dict(sorted(
                    term_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))

                sorted_term_weights_first_15 = {k: sorted_term_weights[k] for k in list(sorted_term_weights.keys())[:weight_thresh]}
                if rank <= rank_thresh:
                    results["gold_docs_analysis"].append({
                        "content": hit_content,
                        "rank": rank,
                        "score": score,
                        "term_weights": sorted_term_weights_first_15
                    })

        return results

    except Exception as e:
        print(f"搜索分析失败: {e}")
        return {
            "query": query_str,
            "error": str(e),
            "gold_docs_analysis": []
        }

