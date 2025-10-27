import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Literal
import os


def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Read JSON file and return data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_recall_values(data_list: List[Dict[str, Any]]) -> Dict[int, float]:
    """Extract Recall@k values from data list"""
    recall_values = {}

    for item in data_list:
        for key, value in item.items():
            if key.startswith('Recall@'):
                k = int(key.split('@')[1])
                if k not in recall_values:
                    recall_values[k] = []
                recall_values[k].append(value)

    # Calculate average for each k value
    recall_avg = {}
    for k in sorted(recall_values.keys()):
        recall_avg[k] = np.mean(recall_values[k])

    return recall_avg


def plot_recall_comparison(recall_dicts: List[Dict[int, float]],
                           labels: List[str],
                           chart_type: Literal['line', 'bar'] = 'line',
                           colors: List[str] = None):
    """Plot Recall comparison between multiple files with choice of chart type"""

    if colors is None:
        # Default color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    plt.figure(figsize=(14, 8))

    # Use the first file's k values as reference (assuming all have same k values)
    k_values = sorted(recall_dicts[0].keys())
    x_positions = np.arange(len(k_values))

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers for each line

    if chart_type == 'line':
        # Plot line chart for each file
        for i, (recall_dict, label) in enumerate(zip(recall_dicts, labels)):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # Convert to percentage
            avg_values = [recall_dict[k] * 100 for k in k_values]

            plt.plot(x_positions, avg_values, f'{marker}-', linewidth=2.5, markersize=10,
                     label=label, color=color, alpha=0.8, markeredgecolor='white', markeredgewidth=1)

    elif chart_type == 'bar':
        # Plot bar chart for each file
        bar_width = 0.8 / len(recall_dicts)  # Dynamic width based on number of datasets

        for i, (recall_dict, label) in enumerate(zip(recall_dicts, labels)):
            color = colors[i % len(colors)]

            # Convert to percentage
            avg_values = [recall_dict[k] * 100 for k in k_values]

            # Calculate positions for each bar group
            positions = x_positions + (i - len(recall_dicts) / 2 + 0.5) * bar_width

            plt.bar(positions, avg_values, bar_width, label=label, color=color, alpha=0.8)

    # Set chart properties
    plt.xlabel('k value (Recall@k)', fontsize=12)
    plt.ylabel('Average Recall Value (%)', fontsize=12)

    chart_title = 'Recall@k Average Comparison'
    if chart_type == 'bar':
        chart_title += ' - Bar Chart'
    else:
        chart_title += ' - Line Chart'
    plt.title(chart_title, fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Set x-axis ticks with proper spacing
    plt.xticks(x_positions, [f'@{k}' for k in k_values], rotation=45)

    # Add value annotations
    for i, k in enumerate(k_values):
        for j, recall_dict in enumerate(recall_dicts):
            value = recall_dict[k] * 100  # Convert to percentage

            if chart_type == 'line':
                # For line charts, position annotations above the points with better alignment
                vertical_offset = 1.5 + (j * 2)  # Smaller offset for percentage values
                color = colors[j % len(colors)]
                plt.annotate(f'{value:.1f}%',
                             (x_positions[i], value),
                             textcoords="offset points",
                             xytext=(0, vertical_offset),
                             ha='center',
                             fontsize=9,
                             color=color,
                             fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor='none'))

            elif chart_type == 'bar':
                # For bar charts, position annotations above the bars
                bar_height = value
                color = colors[j % len(colors)]
                bar_width = 0.8 / len(recall_dicts)
                x_position = x_positions[i] + (j - len(recall_dicts) / 2 + 0.5) * bar_width

                plt.annotate(f'{value:.1f}%',
                             (x_position, bar_height),
                             textcoords="offset points",
                             xytext=(0, 3),
                             ha='center',
                             fontsize=8,
                             color=color,
                             fontweight='bold')

    # Set y-axis to show percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Adjust y-axis limits to accommodate annotations
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max * 1.1)  # Add 10% headroom for annotations

    # Adjust layout
    plt.tight_layout()
    plt.show()


def print_statistics(recall_dicts: List[Dict[int, float]], labels: List[str]):
    """Print statistics for multiple files in percentage format"""
    print("\n" + "=" * 60)
    print("Recall Statistics Comparison (Percentage)")
    print("=" * 60)

    # Print header
    header = "k-value\t\t" + "\t".join(labels)
    print(header)
    print("-" * (len(header) + 20))

    # Print data for each k value
    k_values = sorted(recall_dicts[0].keys())
    for k in k_values:
        row = f"Recall@{k}"
        for recall_dict in recall_dicts:
            percentage_value = recall_dict[k] * 100
            row += f"\t{percentage_value:6.2f}%"
        print(row)


def get_file_name(file_path: str) -> str:
    """Extract file name from path for labeling"""
    return os.path.basename(file_path).replace('.json', '')


def main():
    # File paths list - modify this list as needed
    file_paths = [
        # "outputs/aliyun_2wiki/retrival_results/2wiki_result_youtu_noagent.json",
        # "outputs/aliyun_2wiki/retrival_results/2wiki_result_youtu_agent.json",
        # "outputs/aliyun_2wiki/retrival_results/dpr_example_retrieval_results.json",
        # "outputs/aliyun_2wiki/retrival_results/dpr_plus_example_retrieval_results.json",
        "outputs/aliyun_2wiki/retrival_results/example_retrieval_results_default.json",
        "outputs/aliyun_2wiki/retrival_results/example_retrieval_results_top20.json",
        "outputs/aliyun_2wiki/retrival_results/example_retrieval_results_top40.json",
        "outputs/aliyun_2wiki/retrival_results/example_retrieval_results_top60.json",
        # "outputs/aliyun_isereal/retrival_results/before_refine_dpr_example_retrieval_results.json",
        # "outputs/aliyun_isereal/retrival_results/before_refine_example_retrieval_results.json",
        # "outputs/aliyun_isereal/retrival_results/es_retrieve_res.json",
        # Add more file paths here as needed
    ]

    # Labels for each file (if not provided, will use file names)
    labels = [
        # "2wiki DPR",
        # "2wiki DPR plus",
        "2wiki top 5",
        "2wiki top 20",
        "2wiki top 40",
        "2wiki top 60"
        # "HippoRAG refined REFINED",
        # "Embedding compare RAW question",
        # "HippoRAG refined RAW question",
        # "Elastic Search",
        # Add more labels here corresponding to file_paths
    ]

    # Colors for each line in the plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Chart type selection
    print("Select chart type:")
    print("1. Line Chart")
    print("2. Bar Chart")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "2":
        chart_type = "bar"
    else:
        chart_type = "line"

    try:
        # Read all JSON files
        all_data = []
        recall_averages = []

        print("\nReading files:")
        print("-" * 40)

        for i, file_path in enumerate(file_paths):
            data = read_json_file(file_path)
            all_data.append(data)
            recall_avg = extract_recall_values(data)
            recall_averages.append(recall_avg)

            # Use provided label or generate from file name
            if i < len(labels):
                label = labels[i]
            else:
                label = get_file_name(file_path)

            print(f"{label}: {len(data)} items")

        # Print statistics
        print_statistics(recall_averages, labels[:len(file_paths)])

        # Plot comparison
        plot_recall_comparison(recall_averages, labels[:len(file_paths)], chart_type, colors)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()