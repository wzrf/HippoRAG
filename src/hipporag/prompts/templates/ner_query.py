ner_system = """You're a very effective entity extraction system.
"""

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.
如果问题是中文的话，输出中文；如果问题是英文的话，输出英文。

Question: Which magazine was started first Arthur's Magazine or First for Women?
"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""
# query_prompt_template = """
# Question: {}

# """
prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "Question: ${query}"}
]