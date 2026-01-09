import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
computing_power = [1348460, 1125582, 753576, 377257, 186995, 0]
recall_rate = [0.6042, 0.5819, 0.5444, 0.4889, 0.4472, 0.376]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(computing_power, recall_rate, 'o-', linewidth=2, markersize=8,
         color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=1)

# 设置标题和标签
plt.title('算力与召回率关系图', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('算力', fontsize=12)
plt.ylabel('召回率', fontsize=12)

# 设置网格
plt.grid(True, alpha=0.3, linestyle='--')

# 设置坐标轴范围
plt.xlim(-50000, max(computing_power) + 50000)
plt.ylim(0.35, 0.65)

# 在数据点旁边添加数值标签
for i, (x, y) in enumerate(zip(computing_power, recall_rate)):
    plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points",
                 xytext=(0,10), ha='center', fontsize=9)

# 格式化x轴标签
plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.show()