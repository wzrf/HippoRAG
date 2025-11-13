import json
import re
from openai import OpenAI
from typing import List, Dict, Any


class AliBailianClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        """
        初始化阿里云百炼客户端

        Args:
            api_key: API密钥
            base_url: 服务地址
            model: 模型名称
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def split_question_and_assign_docs(self, question: str, gold_docs: List[str]) -> List[Dict[str, Any]]:
        """
        使用大模型拆分问题并为每个子问题分配对应的gold_docs

        Args:
            question: 原始问题
            gold_docs: 对应的参考文档列表

        Returns:
            包含子问题及其对应gold_docs的字典列表
        """
        # 构建提示词
        prompt = self._build_split_prompt(question, gold_docs)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的问题分析助手，擅长将复杂问题拆分成多个逻辑相关的子问题，并为每个子问题匹配合适的参考文档。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=3000
            )

            # 解析响应
            result = response.choices[0].message.content.strip()
            sub_questions_with_docs = self._parse_response(result, gold_docs)

            return sub_questions_with_docs

        except Exception as e:
            print(f"调用大模型失败: {e}")
            # 如果拆分失败，返回原问题和所有gold_docs作为保底
            return [{
                "question": question,
                "gold_docs": gold_docs
            }]

    def _build_split_prompt(self, question: str, gold_docs: List[str]) -> str:
        """构建拆分问题并分配文档的提示词"""

        # 为每个文档创建编号和摘要
        docs_with_index = []
        for i, doc in enumerate(gold_docs):
            # 提取标题（如果有）
            title_match = re.search(r'标题:\s*(.*?)\n', doc)
            title = title_match.group(1) if title_match else f"文档{i + 1}"

            # 提取内容前300字符作为预览
            content_preview = doc[:300] + "..." if len(doc) > 300 else doc

            docs_with_index.append({
                "index": i,
                "title": title,
                "preview": content_preview
            })

        # 构建文档列表字符串
        docs_list_str = ""
        for doc_info in docs_with_index:
            docs_list_str += f"文档{doc_info['index']}: {doc_info['title']}\n"
            docs_list_str += f"内容预览: {doc_info['preview']}\n\n"

        prompt = f"""
请将以下复杂问题拆分成3-5个逻辑相关的子问题，并为每个子问题分配最相关的参考文档。

原始问题：
{question}

参考文档列表（每个文档前有编号）：
{docs_list_str}

任务要求：
1. 将原始问题拆分成3-5个逻辑相关的子问题
2. 每个子问题应该聚焦于原始问题的一个特定方面
3. 为每个子问题分配最相关的参考文档（使用文档编号，如[0]、[1]等）
4. 确保拆分后的子问题组合起来能够完整覆盖原始问题的所有信息点

返回格式要求（严格的JSON格式）：
{{
    "sub_questions": [
        {{
            "question": "子问题1",
            "doc_indexes": [0, 1]  // 相关的文档编号列表
        }},
        {{
            "question": "子问题2", 
            "doc_indexes": [2]
        }}
    ]
}}

请确保：
- 每个子问题都有对应的doc_indexes数组
- doc_indexes中的数字必须是有效的文档编号（0到{len(gold_docs) - 1}）
- 如果某个子问题与多个文档相关，可以包含多个编号
- 直接返回JSON格式，不要有其他内容
"""
        return prompt

    def _parse_response(self, response: str, gold_docs: List[str]) -> List[Dict[str, Any]]:
        """解析大模型的响应，提取子问题和对应的文档"""
        try:
            # 尝试直接解析JSON
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            if isinstance(data, dict) and "sub_questions" in data:
                sub_questions_data = data["sub_questions"]
                result = []

                for item in sub_questions_data:
                    if isinstance(item, dict) and "question" in item:
                        question = item["question"]
                        doc_indexes = item.get("doc_indexes", [])

                        # 验证文档索引的有效性
                        valid_doc_indexes = []
                        for idx in doc_indexes:
                            if isinstance(idx, int) and 0 <= idx < len(gold_docs):
                                valid_doc_indexes.append(idx)

                        # 如果没有有效的文档索引，使用所有文档作为保底
                        if not valid_doc_indexes:
                            valid_doc_indexes = list(range(len(gold_docs)))

                        # 根据索引获取对应的文档
                        assigned_docs = [gold_docs[i] for i in valid_doc_indexes]

                        result.append({
                            "question": question,
                            "gold_docs": assigned_docs
                        })

                return result if result else [{"question": response, "gold_docs": gold_docs}]

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")

        # 如果解析失败，返回原始响应作为单个问题
        return [{"question": response, "gold_docs": gold_docs}]


def process_json_file(input_file: str, output_file: str, api_key: str, base_url: str, model: str):
    """
    处理JSON文件，拆分所有问题并为每个子问题分配对应的gold_docs

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        api_key: API密钥
        base_url: 服务地址
        model: 模型名称
    """
    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return


    # 初始化客户端
    client = AliBailianClient(api_key, base_url, model)

    # 处理每个问题
    processed_data = []
    total_items = len(data)

    for i, item in enumerate(data, 1):
        print(f"处理进度: {i}/{total_items}")

        original_question = item.get("question", "")
        gold_docs = item.get("gold_docs", [])

        if not original_question:
            print(f"第 {i} 项缺少问题，跳过")
            continue

        print(f"原始问题: {original_question[:100]}...")
        print(f"参考文档数量: {len(gold_docs)}")

        # 拆分问题并分配文档
        sub_questions_with_docs = client.split_question_and_assign_docs(original_question, gold_docs)

        # 为每个子问题创建新条目
        for sub_item in sub_questions_with_docs:
            new_item = {
                "question": sub_item["question"],
                "gold_docs": sub_item["gold_docs"],
                "original_question": original_question
            }
            processed_data.append(new_item)

        print(f"原问题拆分为 {len(sub_questions_with_docs)} 个子问题")

        # 显示拆分结果示例
        for j, sub_item in enumerate(sub_questions_with_docs):
            print(f"  子问题 {j + 1}: {sub_item['question'][:80]}...")
            print(f"    分配文档数: {len(sub_item['gold_docs'])}")

        print("-" * 80)

    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"处理完成！共生成 {len(processed_data)} 个问题，已保存到: {output_file}")

        # 输出统计信息
        original_count = len(data)
        sub_question_count = len(processed_data)
        avg_split = sub_question_count / original_count if original_count > 0 else 0
        print(
            f"统计信息: 原始问题 {original_count} 个，拆分后问题 {sub_question_count} 个，平均每个问题拆分 {avg_split:.2f} 个子问题")

    except Exception as e:
        print(f"保存文件失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 请在这里填写您的配置信息
    API_KEY = "sk-0ec0ef1cfb43447a8cca4a6fe05519fa"  # 替换为您的API密钥
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 替换为您的服务地址
    MODEL = "deepseek-v3"  # 替换为您的模型名称

    # 文件路径
    INPUT_JSON_FILE = "/Users/xumengyao/work/QIYUAN/HippoRAG/outputs/aliyun_isereal/multi_hop/refined_questions_format.json"  # 输入JSON文件路径
    OUTPUT_JSON_FILE = "/Users/xumengyao/work/QIYUAN/HippoRAG/outputs/aliyun_isereal/multi_hop/refined_questions_format_split.json"  # 输出JSON文件路径

    # 处理文件
    process_json_file(INPUT_JSON_FILE, OUTPUT_JSON_FILE, API_KEY, BASE_URL, MODEL)