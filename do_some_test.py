import os, json
import dashscope
from http import HTTPStatus
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from openai import OpenAI
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class OnlineEncoder:
    def __init__(self):
        self.embedding_model_name = 'text-embedding-v4'
        llm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        llm_api_key = os.environ.get("DASHSCOPE_API_KEY")

        self.client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
        )

    def get_sentence_embedding_dimension(self):
        return 1024

    def encode(self, text, batch_size=10, convert_to_tensor=False, device=None):
        """
        对文本进行嵌入编码
        """
        is_single_string = isinstance(text, str)

        if is_single_string:
            text = [text]

        all_embeddings = []

        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i + batch_size]

            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.embedding_model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Embedding计算失败: {e}")
                # 返回随机向量作为fallback
                batch_embeddings = [np.random.randn(1024).tolist() for _ in batch_texts]
                all_embeddings.extend(batch_embeddings)

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        if is_single_string:
            result = embeddings_array[0]
        else:
            result = embeddings_array

        if convert_to_tensor:
            return torch.tensor(result).detach()
        else:
            return result


def call_rerank(query: str, texts: list[str]):
    """
    调用阿里云的rerank服务
    """
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=texts,
        return_documents=True
    )

    if resp.status_code == HTTPStatus.OK:
        final_res = {}
        print(f"mengyao_debug query is {query}")
        for result in resp.output['results']:
            print(f"排名: {result['index']}, 相关性得分: {result['relevance_score']:.4f} "
                  f"文档内容: {result['document']['text']}")
            final_res[result['document']['text']] = result['relevance_score']
        return final_res
    else:
        print(f"请求失败，状态码: {resp.status_code}, 错误信息: {resp.message}")
        return {}


def calculate_cosine_similarity(embedding1, embedding2):
    """
    计算两个嵌入向量的余弦相似度
    """
    if isinstance(embedding1, np.ndarray):
        embedding1 = torch.tensor(embedding1)
    if isinstance(embedding2, np.ndarray):
        embedding2 = torch.tensor(embedding2)

    # 确保向量是2D的
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)

    # 计算余弦相似度
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item() if similarity.numel() == 1 else similarity.numpy()


def get_user_choice():
    """
    获取用户选择：分析chunk还是fact
    """
    print("请选择分析模式：")
    print("1. chunk - 分析文档块")
    print("2. fact - 分析事实")

    while True:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == '1':
            return 'chunk'
        elif choice == '2':
            return 'fact'
        else:
            print("输入无效，请重新输入")


def extract_data_by_mode(data, mode):
    """
    根据模式提取数据
    """
    if mode == 'chunk':
        # 提取chunk数据
        gold_items = data.get('chunk_ids', [])
        all_items = []
        item_id_to_content = {}

        for chunk in data.get('chunks_list', []):
            chunk_id = chunk.get('hash_id', '')
            content = chunk.get('content', '')
            # 取前200个字符作为代表性内容
            short_content = content[:200] + "..." if len(content) > 200 else content
            all_items.append(short_content)
            item_id_to_content[chunk_id] = short_content

        # 将gold item IDs转换为内容
        gold_contents = []
        for item_id in gold_items:
            if item_id in item_id_to_content:
                gold_contents.append(item_id_to_content[item_id])
            else:
                print(f"警告: gold chunk '{item_id}' 不在chunks_list中")

        return gold_contents, all_items, item_id_to_content

    else:  # fact模式
        # 提取fact数据
        gold_items = data.get('facts_referenced', [])
        all_items = []

        # 从total_fact_list中提取所有facts
        total_fact_list = data.get('total_fact_list', {})
        for chunk_facts in total_fact_list.values():
            all_items.extend(chunk_facts)

        # 去重
        all_items = list(set(all_items))

        return gold_items, all_items, {}


def run_comparison_analysis():
    """
    主分析函数，支持chunk和fact两种模式
    """
    # 获取用户选择
    mode = get_user_choice()
    print(f"选择的分析模式: {mode}")

    questions = []
    file_data = []

    # 初始化embedding编码器
    encoder = OnlineEncoder()

    # 读取数据
    for filename in os.listdir("outputs/aliyun_isereal/multi_hop"):
        if "multi_hop_question" in filename:
            with open(f"outputs/aliyun_isereal/multi_hop/{filename}", 'r', encoding='utf-8') as f:
                data = json.load(f)
                questions.append(data["question"])
                file_data.append(data)

    print(f"总共问题数: {len(questions)}")

    # 存储结果
    results = []
    # 存储详细的排名信息用于表格输出
    ranking_data = []

    for i, (question, data) in enumerate(zip(questions, file_data)):
        print(f"\n=== 处理第 {i + 1} 个问题 ===")
        print(f"问题: {question}")

        # 根据模式提取数据
        gold_items, all_items, id_to_content = extract_data_by_mode(data, mode)
        print(f"Gold {mode}s: {len(gold_items)}个")
        print(f"所有{mode}s数量: {len(all_items)}")

        if not gold_items:
            print(f"第 {i + 1} 个问题没有gold {mode}s，跳过")
            continue

        if not all_items:
            print(f"第 {i + 1} 个问题没有{mode}s，跳过")
            continue

        # 调用rerank
        rerank_scores = call_rerank(question, all_items)

        if not rerank_scores:
            print(f"第 {i + 1} 个问题rerank失败，跳过")
            continue

        # 计算rerank排名百分比
        rerank_ranking = calculate_ranking_percentile(rerank_scores, gold_items, all_items, "rerank")

        # 计算gold items的得分统计
        gold_scores = []
        for gold_item in gold_items:
            if gold_item in rerank_scores:
                gold_scores.append(rerank_scores[gold_item])
            else:
                print(f"警告: gold {mode} 不在rerank结果中")

        if not gold_scores:
            print(f"第 {i + 1} 个问题没有找到对应的gold {mode}得分")
            continue

        avg_gold_score = sum(gold_scores) / len(gold_scores)
        min_gold_score = min(gold_scores) if gold_scores else 0
        max_gold_score = max(gold_scores) if gold_scores else 0

        # 计算所有items的平均得分（不包括gold items）
        other_items_scores = []
        for item, score in rerank_scores.items():
            if item not in gold_items:
                other_items_scores.append(score)

        if not other_items_scores:
            print(f"第 {i + 1} 个问题没有其他{mode}s")
            continue

        avg_other_score = sum(other_items_scores) / len(other_items_scores)

        # 判断gold items得分是否高于其他items
        is_higher = avg_gold_score > avg_other_score

        # 计算embedding相似度
        embedding_results = calculate_embedding_similarity(encoder, question, gold_items, all_items)

        # 计算embedding排名百分比
        embedding_ranking = calculate_ranking_percentile(
            embedding_results['all_similarities'],
            gold_items,
            all_items,
            "embedding"
        )

        # 存储结果
        result = {
            'question_index': i,
            'question': question,
            'mode': mode,
            'avg_gold_score': avg_gold_score,
            'min_gold_score': min_gold_score,
            'max_gold_score': max_gold_score,
            'gold_scores': gold_scores,
            'avg_other_score': avg_other_score,
            'other_scores': other_items_scores,
            'is_gold_higher': is_higher,
            'embedding_results': embedding_results,
            'rerank_ranking': rerank_ranking,
            'embedding_ranking': embedding_ranking
        }
        results.append(result)

        # 为表格收集数据
        for j, gold_item in enumerate(gold_items):
            rerank_percentile = rerank_ranking['gold_item_percentiles'].get(gold_item, 'N/A')
            embedding_percentile = embedding_ranking['gold_item_percentiles'].get(gold_item, 'N/A')

            ranking_data.append({
                'mode': mode,
                'question_index': i,
                'question': question[:100] + "..." if len(question) > 100 else question,
                'gold_item': gold_item[:100] + "..." if len(gold_item) > 100 else gold_item,
                'rerank_score': rerank_scores.get(gold_item, 0),
                'rerank_percentile': rerank_percentile,
                'embedding_similarity': embedding_results['all_similarities'].get(gold_item, 0),
                'embedding_percentile': embedding_percentile,
                'total_items_count': len(all_items)
            })

        print(
            f"Gold {mode}s得分统计 - 平均: {avg_gold_score:.4f}, 最低: {min_gold_score:.4f}, 最高: {max_gold_score:.4f}")
        print(f"其他{mode}s平均得分: {avg_other_score:.4f}")
        print(f"Gold {mode}s得分是否更高: {is_higher}")
        print(
            f"Embedding相似度 - Gold {mode}s平均: {embedding_results['avg_gold_similarity']:.4f}, 其他{mode}s平均: {embedding_results['avg_other_similarity']:.4f}")
        print(f"Rerank排名百分比: {rerank_ranking['gold_item_percentiles']}")
        print(f"Embedding排名百分比: {embedding_ranking['gold_item_percentiles']}")

    # 可视化结果
    plot_results(results, mode)

    # 生成排名表格
    generate_ranking_table(ranking_data, mode)

    # 保存详细结果
    save_detailed_results(results, mode)


def calculate_ranking_percentile(scores_dict, gold_items, all_items, ranking_type):
    """
    计算gold items在所有items中的排名百分比 (0-1, 1表示最好)
    """
    # 按得分降序排序
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    # 创建排名百分比字典
    ranking_percentiles = {}
    total_items = len(sorted_items)

    for rank, (item, score) in enumerate(sorted_items, 1):

        percentile = rank / total_items
        ranking_percentiles[item] = percentile

    # 提取gold items的排名百分比
    gold_item_percentiles = {}
    for gold_item in gold_items:
        gold_item_percentiles[gold_item] = ranking_percentiles.get(gold_item, 'N/A')

    # 计算平均排名百分比
    valid_percentiles = [p for p in gold_item_percentiles.values() if p != 'N/A']
    avg_percentile = sum(valid_percentiles) / len(valid_percentiles) if valid_percentiles else 0

    return {
        'gold_item_percentiles': gold_item_percentiles,
        'avg_percentile': avg_percentile,
        'best_percentile': max(valid_percentiles) if valid_percentiles else 'N/A',
        'worst_percentile': min(valid_percentiles) if valid_percentiles else 'N/A',
        'total_items': len(all_items)
    }


def calculate_embedding_similarity(encoder, question, gold_items, all_items):
    """
    计算embedding相似度
    """
    try:
        # 计算问题的embedding
        question_embedding = encoder.encode(question, convert_to_tensor=True)

        # 计算所有items的embedding相似度
        all_similarities = {}
        for item in all_items:
            item_embedding = encoder.encode(item, convert_to_tensor=True)
            similarity = calculate_cosine_similarity(question_embedding, item_embedding)
            all_similarities[item] = similarity

        # 计算gold items的平均相似度
        gold_similarities = [all_similarities[item] for item in gold_items if item in all_similarities]
        avg_gold_similarity = sum(gold_similarities) / len(gold_similarities) if gold_similarities else 0

        # 计算其他items的平均相似度
        other_similarities = [all_similarities[item] for item in all_items if item not in gold_items]
        avg_other_similarity = sum(other_similarities) / len(other_similarities) if other_similarities else 0

        return {
            'all_similarities': all_similarities,
            'avg_gold_similarity': avg_gold_similarity,
            'avg_other_similarity': avg_other_similarity,
            'gold_similarities': gold_similarities,
            'other_similarities': other_similarities,
            'is_gold_higher_embedding': avg_gold_similarity > avg_other_similarity
        }
    except Exception as e:
        print(f"Embedding相似度计算失败: {e}")
        all_similarities = {item: 0 for item in all_items}
        return {
            'all_similarities': all_similarities,
            'avg_gold_similarity': 0,
            'avg_other_similarity': 0,
            'gold_similarities': [],
            'other_similarities': [],
            'is_gold_higher_embedding': False
        }


def generate_ranking_table(ranking_data, mode):
    """
    生成详细的排名表格
    """
    if not ranking_data:
        print("没有排名数据可展示")
        return

    # 创建DataFrame
    df = pd.DataFrame(ranking_data)

    # 过滤掉无效百分比
    df_valid = df[(df['rerank_percentile'] != 'N/A') & (df['embedding_percentile'] != 'N/A')].copy()

    if len(df_valid) == 0:
        print("没有有效的排名百分比数据")
        return

    # 转换数据类型
    df_valid['rerank_percentile'] = df_valid['rerank_percentile'].astype(float)
    df_valid['embedding_percentile'] = df_valid['embedding_percentile'].astype(float)

    # 计算平均百分比
    df_valid['avg_percentile'] = (df_valid['rerank_percentile'] + df_valid['embedding_percentile']) / 2

    # 保存详细表格到CSV
    output_filename = f'gold_{mode}s_percentile_ranking.csv'
    df_valid.to_csv(output_filename, index=False, encoding='utf-8-sig')

    # 生成汇总统计
    summary_stats = df_valid.groupby('question_index').agg({
        'rerank_percentile': ['mean', 'min', 'max'],
        'embedding_percentile': ['mean', 'min', 'max'],
        'avg_percentile': 'mean',
        'gold_item': 'count'
    }).round(3)

    # 重命名列
    summary_stats.columns = [
        'rerank_avg_percentile', 'rerank_min_percentile', 'rerank_max_percentile',
        'embedding_avg_percentile', 'embedding_min_percentile', 'embedding_max_percentile',
        'avg_percentile',
        f'gold_{mode}_count'
    ]

    summary_stats.to_csv(f'gold_{mode}s_percentile_summary.csv', encoding='utf-8-sig')

    # 打印排名概览
    print(f"\n=== Gold {mode}s排名百分比概览 ===")
    print(f"总Gold {mode}数量: {len(df_valid)}")
    print(f"平均Rerank排名百分比: {df_valid['rerank_percentile'].mean():.3f}")
    print(f"平均Embedding排名百分比: {df_valid['embedding_percentile'].mean():.3f}")
    print(f"平均综合排名百分比: {df_valid['avg_percentile'].mean():.3f}")

    # 可视化排名分布
    plot_percentile_distribution(df_valid, mode)


def plot_percentile_distribution(df, mode):
    """
    可视化排名百分比分布
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Rerank排名百分比分布
    ax1.hist(df['rerank_percentile'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Rerank排名百分比')
    ax1.set_ylabel(f'Gold {mode}数量')
    ax1.set_title(f'Gold {mode}s的Rerank排名百分比分布')
    ax1.grid(True, alpha=0.3)

    # 2. Embedding排名百分比分布
    ax2.hist(df['embedding_percentile'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Embedding排名百分比')
    ax2.set_ylabel(f'Gold {mode}数量')
    ax2.set_title(f'Gold {mode}s的Embedding排名百分比分布')
    ax2.grid(True, alpha=0.3)

    # 3. 排名百分比对比
    ax3.scatter(df['rerank_percentile'], df['embedding_percentile'], alpha=0.6, color='purple')
    ax3.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    ax3.set_xlabel('Rerank排名百分比')
    ax3.set_ylabel('Embedding排名百分比')
    ax3.set_title(f'Rerank vs Embedding排名百分比对比')
    ax3.grid(True, alpha=0.3)

    # 4. 综合排名百分比分布
    ax4.hist(df['avg_percentile'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.set_xlabel('综合排名百分比')
    ax4.set_ylabel(f'Gold {mode}数量')
    ax4.set_title(f'Gold {mode}s的综合排名百分比分布')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'gold_{mode}s_percentile_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_results(results: List[Dict], mode: str):
    """
    可视化结果
    """
    if not results:
        print("没有有效结果可展示")
        return

    # 准备数据
    indices = [r['question_index'] for r in results]
    gold_scores = [r['avg_gold_score'] for r in results]
    other_scores = [r['avg_other_score'] for r in results]
    min_gold_scores = [r['min_gold_score'] for r in results]
    is_higher = [r['is_gold_higher'] for r in results]

    # Embedding相似度数据
    gold_embedding = [r['embedding_results']['avg_gold_similarity'] for r in results]
    other_embedding = [r['embedding_results']['avg_other_similarity'] for r in results]
    is_higher_embedding = [r['embedding_results']['is_gold_higher_embedding'] for r in results]

    # 排名百分比数据
    rerank_avg_percentiles = [r['rerank_ranking']['avg_percentile'] for r in results]
    embedding_avg_percentiles = [r['embedding_ranking']['avg_percentile'] for r in results]

    # 创建图形 - 四个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 第一个子图：Rerank得分对比
    x = np.arange(len(results))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, gold_scores, width, label=f'Gold {mode}s平均得分', color='green', alpha=0.7)
    bars2 = ax1.bar(x + width / 2, other_scores, width, label=f'其他{mode}s平均得分', color='red', alpha=0.7)

    # 标记哪些问题的gold items得分更高（使用星号替代对勾）
    for i, higher in enumerate(is_higher):
        if higher:
            ax1.text(i, max(gold_scores[i], other_scores[i]) + 0.01, '*',
                     ha='center', va='bottom', fontsize=14, color='blue', fontweight='bold')

    # 添加gold items最低分标记
    for i, (gold_score, min_score) in enumerate(zip(gold_scores, min_gold_scores)):
        ax1.text(i - width / 2, gold_score + 0.005, f'min:{min_score:.3f}',
                 ha='center', va='bottom', fontsize=8, color='darkgreen', rotation=45)

    ax1.set_xlabel('问题索引')
    ax1.set_ylabel('Rerank得分')
    ax1.set_title(f'Rerank得分对比: Gold {mode}s vs 其他{mode}s')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Q{i + 1}' for i in indices])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 第二个子图：Embedding相似度对比
    bars3 = ax2.bar(x - width / 2, gold_embedding, width, label=f'Gold {mode}s相似度', color='lightblue', alpha=0.7)
    bars4 = ax2.bar(x + width / 2, other_embedding, width, label=f'其他{mode}s相似度', color='orange', alpha=0.7)

    # 标记embedding相似度更高的
    for i, higher in enumerate(is_higher_embedding):
        if higher:
            ax2.text(i, max(gold_embedding[i], other_embedding[i]) + 0.01, '*',
                     ha='center', va='bottom', fontsize=14, color='blue', fontweight='bold')

    ax2.set_xlabel('问题索引')
    ax2.set_ylabel('余弦相似度')
    ax2.set_title(f'Embedding相似度对比: Gold {mode}s vs 其他{mode}s')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Q{i + 1}' for i in indices])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 第三个子图：平均排名百分比对比
    ax3.plot(x, rerank_avg_percentiles, 'o-', label='Rerank平均排名百分比', color='green', linewidth=2, markersize=6)
    ax3.plot(x, embedding_avg_percentiles, 's-', label='Embedding平均排名百分比', color='blue', linewidth=2,
             markersize=6)
    ax3.set_xlabel('问题索引')
    ax3.set_ylabel('平均排名百分比 (1=最好)')
    ax3.set_title(f'Gold {mode}s的平均排名百分比对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Q{i + 1}' for i in indices])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 第四个子图：综合统计信息
    higher_count = sum(is_higher)
    lower_count = len(results) - higher_count
    higher_count_embedding = sum(is_higher_embedding)

    categories = ['Rerank更高', 'Embedding更高']
    counts = [higher_count, higher_count_embedding]
    colors = ['lightgreen', 'lightblue']

    bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
    ax4.set_ylabel('问题数量')
    ax4.set_title(f'Gold {mode}s表现更好的问题数量')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count} ({count / len(results) * 100:.1f}%)',
                 ha='center', va='bottom')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'gold_{mode}s_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print(f"\n=== 统计结果 ===")
    print(f"总问题数: {len(results)}")
    print(f"Rerank - Gold {mode}s得分更高: {higher_count} ({higher_count / len(results) * 100:.1f}%)")
    print(
        f"Embedding - Gold {mode}s相似度更高: {higher_count_embedding} ({higher_count_embedding / len(results) * 100:.1f}%)")
    print(f"平均Rerank排名百分比: {np.mean(rerank_avg_percentiles):.3f}")
    print(f"平均Embedding排名百分比: {np.mean(embedding_avg_percentiles):.3f}")


def save_detailed_results(results: List[Dict], mode: str):
    """
    保存详细结果到JSON文件
    """
    output_data = []
    for result in results:
        output_data.append({
            'question_index': result['question_index'],
            'question': result['question'],
            'mode': mode,
            'rerank_scores': {
                'avg_gold_score': result['avg_gold_score'],
                'min_gold_score': result['min_gold_score'],
                'max_gold_score': result['max_gold_score'],
                'avg_other_score': result['avg_other_score'],
                'is_gold_higher': result['is_gold_higher'],
                'score_difference': result['avg_gold_score'] - result['avg_other_score'],
                'gold_item_percentiles': result['rerank_ranking']['gold_item_percentiles'],
                'avg_percentile': result['rerank_ranking']['avg_percentile']
            },
            'embedding_similarity': {
                'avg_gold_similarity': result['embedding_results']['avg_gold_similarity'],
                'avg_other_similarity': result['embedding_results']['avg_other_similarity'],
                'is_gold_higher': result['embedding_results']['is_gold_higher_embedding'],
                'similarity_difference': result['embedding_results']['avg_gold_similarity'] -
                                         result['embedding_results']['avg_other_similarity'],
                'gold_item_percentiles': result['embedding_ranking']['gold_item_percentiles'],
                'avg_percentile': result['embedding_ranking']['avg_percentile']
            }
        })

    with open(f'comprehensive_comparison_results_{mode}.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"详细结果已保存到 'comprehensive_comparison_results_{mode}.json'")


if __name__ == '__main__':
    run_comparison_analysis()