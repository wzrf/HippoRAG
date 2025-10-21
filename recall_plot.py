import json
import matplotlib.pyplot as plt
import numpy as np


def read_json_file(file_path):
    """Read JSON file and return data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_recall_values(data_list):
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


def plot_recall_comparison(recall_dict1, recall_dict2, label1="File 1", label2="File 2"):
    """Plot Recall comparison between two files"""
    plt.figure(figsize=(14, 8))

    # Extract k values and corresponding averages
    k_values = sorted(recall_dict1.keys())
    avg_values1 = [recall_dict1[k] for k in k_values]
    avg_values2 = [recall_dict2[k] for k in k_values]

    # Create x positions with proper spacing
    x_positions = np.arange(len(k_values))

    # Plot line chart
    plt.plot(x_positions, avg_values1, 'o-', linewidth=2.5, markersize=8,
             label=label1, color='blue', alpha=0.8)
    plt.plot(x_positions, avg_values2, 's-', linewidth=2.5, markersize=8,
             label=label2, color='red', alpha=0.8)

    # Set chart properties
    plt.xlabel('k value (Recall@k)', fontsize=12)
    plt.ylabel('Average Recall Value', fontsize=12)
    plt.title('Recall@k Average Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Set x-axis ticks with proper spacing
    plt.xticks(x_positions, [f'@{k}' for k in k_values], rotation=45)

    # Add value annotations with offset to avoid overlap
    for i, (k, v1, v2) in enumerate(zip(k_values, avg_values1, avg_values2)):
        plt.annotate(f'{v1:.3f}', (x_positions[i], v1), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, color='blue', fontweight='bold')
        plt.annotate(f'{v2:.3f}', (x_positions[i], v2), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=9, color='red', fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def print_statistics(recall_dict, label):
    """Print statistics"""
    print(f"\n{label} Recall Statistics:")
    print("-" * 35)
    for k, avg in sorted(recall_dict.items()):
        print(f"Recall@{k}: {avg:.4f}")


def main():
    # File paths
    file1_path = "outputs/aliyun_isereal/retrival_results/example_retrieval_results.json"
    file2_path = "outputs/aliyun_2wiki/retrival_results_bak/example_retrieval_results.json"

    try:
        # Read JSON files
        data1 = read_json_file(file1_path)
        data2 = read_json_file(file2_path)

        print(f"File 1 contains {len(data1)} items")
        print(f"File 2 contains {len(data2)} items")

        # Extract and calculate Recall averages
        recall_avg1 = extract_recall_values(data1)
        recall_avg2 = extract_recall_values(data2)

        # Print statistics
        print_statistics(recall_avg1, "File 1")
        print_statistics(recall_avg2, "File 2")

        # Plot comparison
        plot_recall_comparison(recall_avg1, recall_avg2, "Our question", "2wiki Question")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()