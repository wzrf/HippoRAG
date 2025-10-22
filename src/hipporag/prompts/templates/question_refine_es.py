QA_refine_system = """

您是一个专业的问题优化专家，您需要通过优化问题的表达方式来减少问题中关键词出现的数量。现在这个问题里面的关键词让全文检索变得太容易了。
注意问题优化之后不要变得很短，尽量保持问题的长度，不要缩短问题的长度。
"""

QA_refine_user = """

我会给出问题，以及问题中出现的关键字，以及这些关键字的权重；请优化问题，在***保证含义不变***的情况下，尽量少出现关键字；权重越高的关键越要少出现；

原始问题如下：${question}

关键字如下：${keywords}

请输出json

{
    "question": "优化后的问题",
    "explain": "思考过程",
}
"""

prompt_template = [
    {"role": "system", "content": QA_refine_system},
    {"role": "user", "content": QA_refine_user},
]