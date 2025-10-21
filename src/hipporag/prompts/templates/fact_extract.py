QA_fact_extract_entities_system = """

你是一个事实提取专家。你的任务是从提供的文本chunks中提取所有事实性信息。
"""
QA_fact_extract_entities_user = """

每个chunk包含一个唯一的hash_id和content内容。请严格提取事实，包括具体事件（发生了什么、时间、日期，具体到年月日，）、人物行动（谁做了什么、谁说了什么、什么事情发生了）、职位任命（谁担任了什么职位）等明确发生的客观信息。

因为每一条fact会被单独使用，所以每一个fact都要单独标注日期、主语、宾语要完善不要没头没尾

避免提取观点、辩论、分析或评价性内容。

输出必须是一个JSON对象，其中每个键是chunk的hash_id，值是一个字符串数组，包含提取的事实。每个事实应简洁但完整，包含关键细节如日期。示例输出格式：

{"chunk-2d7ebd5a1c92eabd748c8edad387f8c2": ["事实1", "事实2"]}。请基于输入的chunks列表进行处理。

我给出的文本如下：

"""



prompt_template = [
    {"role": "system", "content": QA_fact_extract_entities_system},
    {"role": "user", "content": QA_fact_extract_entities_user + "${passage}"},
]