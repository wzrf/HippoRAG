import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
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
                           colors: List[str] = None):
    """Plot Recall comparison between multiple files"""

    if colors is None:
        # Default color scheme
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    plt.figure(figsize=(14, 8))

    # Use the first file's k values as reference (assuming all have same k values)
    k_values = sorted(recall_dicts[0].keys())
    x_positions = np.arange(len(k_values))

    # Plot line chart for each file
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers for each line

    for i, (recall_dict, label) in enumerate(zip(recall_dicts, labels)):
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]

        if i >= len(markers):
            marker = markers[i % len(markers)]
        else:
            marker = markers[i]

        avg_values = [recall_dict[k] for k in k_values]

        plt.plot(x_positions, avg_values, f'{marker}-', linewidth=2.5, markersize=8,
                 label=label, color=color, alpha=0.8)

    # Set chart properties
    plt.xlabel('k value (Recall@k)', fontsize=12)
    plt.ylabel('Average Recall Value', fontsize=12)
    plt.title('Recall@k Average Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Set x-axis ticks with proper spacing
    plt.xticks(x_positions, [f'@{k}' for k in k_values], rotation=45)

    # Add value annotations (optional - can be commented out if too crowded)
    for i, k in enumerate(k_values):
        for j, recall_dict in enumerate(recall_dicts):
            value = recall_dict[k]
            vertical_offset = 10 + (j * 25)  # Stagger annotations to avoid overlap
            color = colors[j % len(colors)]
            plt.annotate(f'{value:.3f}', (x_positions[i], value),
                         textcoords="offset points", xytext=(0, vertical_offset),
                         ha='center', fontsize=8, color=color, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def print_statistics(recall_dicts: List[Dict[int, float]], labels: List[str]):
    """Print statistics for multiple files"""
    print("\n" + "=" * 50)
    print("Recall Statistics Comparison")
    print("=" * 50)

    # Print header
    header = "k-value\t" + "\t".join(labels)
    print(header)
    print("-" * (len(header) + 20))

    # Print data for each k value
    k_values = sorted(recall_dicts[0].keys())
    for k in k_values:
        row = f"Recall@{k}"
        for recall_dict in recall_dicts:
            row += f"\t{recall_dict[k]:.4f}"
        print(row)


def get_file_name(file_path: str) -> str:
    """Extract file name from path for labeling"""
    return os.path.basename(file_path).replace('.json', '')


def main():
    # File paths list - modify this list as needed
    file_paths = [
        "outputs/aliyun_isereal/retrival_results/dpr_example_retrieval_results.json",
        "outputs/aliyun_isereal/retrival_results/example_retrieval_results.json",
        "/Users/xumengyao/PycharmProjects/test/recall_analysis_results/detailed_recall_results.json",
        # Add more file paths here as needed
    ]

    # Labels for each file (if not provided, will use file names)
    labels = [
        "Embedding compare",
        "HippoRAG",
        "Elastic Search",
        # Add more labels here corresponding to file_paths
    ]

    # Colors for each line in the plot
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    try:
        # Read all JSON files
        all_data = []
        recall_averages = []

        print("Reading files:")
        print("-" * 30)

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
        plot_recall_comparison(recall_averages, labels[:len(file_paths)], colors)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()