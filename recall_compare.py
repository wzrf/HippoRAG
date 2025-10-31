import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Node Similarity', 'Text Similarity', 'Node Recall', 'Text Recall']
raw_data = [0.32378, 0.35841, 0.19529, 0.05853]
refined_data = [0.32945, 0.35982, 0.17090, 0.05574]

# 计算差异
differences = [refined - raw for raw, refined in zip(raw_data, refined_data)]

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 第一个子图：并排条形图
x = np.arange(len(categories))
width = 0.35

bars1 = ax1.bar(x - width/2, raw_data, width, label='Raw Question', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x + width/2, refined_data, width, label='Refined Question', alpha=0.8, color='lightcoral')

ax1.set_xlabel('Metrics')
ax1.set_ylabel('Scores')
ax1.set_title('Comparison: Raw vs Refined Questions')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45)
ax1.legend()

# 在条形上添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 第二个子图：差异图
colors = ['green' if diff > 0 else 'red' for diff in differences]
bars3 = ax2.bar(categories, differences, color=colors, alpha=0.7)

ax2.set_xlabel('Metrics')
ax2.set_ylabel('Difference (Refined - Raw)')
ax2.set_title('Performance Differences')
ax2.set_xticklabels(categories, rotation=45)

# 在差异条形上添加数值标签
for bar in bars3:
    height = bar.get_height()
    va = 'bottom' if height > 0 else 'top'
    color = 'darkgreen' if height > 0 else 'darkred'
    ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.001),
             f'{height:+.3f}', ha='center', va=va, fontsize=10, color=color, weight='bold')

# 添加零线
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# 打印详细分析
print("\n=== 详细分析 ===")
for i, category in enumerate(categories):
    diff = differences[i]
    change = "提升" if diff > 0 else "下降"
    print(f"{category}: {change} {abs(diff):.4f} "
          f"({raw_data[i]:.3f} → {refined_data[i]:.3f})")

print(f"\n总结:")
print(f"- Node Similarity 和 Text Similarity 略有提升")
print(f"- Node Recall 和 Text Recall 略有下降")
print(f"- 整体变化幅度较小")

# [raw_question_default] node similarity is 0.3237795260061958, text similarity is 0.35840736227038933, average_node_recall_scores is 0.19529252164943295 average_text_recall_scores is 0.058534222982847814
# [refined_question_default] node similarity is 0.32944642261292895, text similarity is 0.35981611781149303, average_node_recall_scores is 0.17089549407477572 average_text_recall_scores is 0.05573808219155889