# generate_comparison_plots.py
import pandas as pd
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# --- Plotting Style Configuration ---
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12

''' 
NOTE: This was made with the HELP of ChatGPT.

The only functionality is combining the reports into a single comparison plot, nothing more.
'''

def load_and_process_reports(reports_dir: Path) -> pd.DataFrame:
    """
    Finds all 'report_metrics.json' files in the analysis subdirectories,
    loads them, and processes them into a single tidy DataFrame.
    """
    all_pipeline_data = []

    report_files = list(reports_dir.glob('analysis_*/report_metrics.json'))
    if not report_files:
        raise FileNotFoundError(f"No 'report_metrics.json' files found in subdirectories of '{reports_dir}'. "
                                "Please ensure you have run the analyzer script first.")
    print(f"Found {len(report_files)} report files to process.")

    for report_file in report_files:
        dir_name = report_file.parent.name
        try:
            # Flexible name parsing to handle 'raw_llm', 'combination', and single names like 'svm'
            name_parts = dir_name.split('_')
            if "raw" in name_parts and "llm" in name_parts:
                pipeline_name = "RAW LLM"
            elif name_parts[1].lower() == 'combination':
                pipeline_name = 'HYBRID'
            else:
                pipeline_name = name_parts[1].upper()
        except IndexError:
            print(f"Warning: Could not parse pipeline name from '{dir_name}'. Skipping.")
            continue

        with open(report_file, 'r') as f:
            data = json.load(f)

        by_category = data.get('by_category', {})
        for category, metrics in by_category.items():
            all_pipeline_data.append({
                'pipeline': pipeline_name,
                'category': category,
                'tpr': metrics.get('true_positive_rate', 0.0),
                'fpr': metrics.get('false_positive_rate', 0.0)
            })

    if not all_pipeline_data:
        raise ValueError("No data could be processed. Check your report files and directory structure.")

    return pd.DataFrame(all_pipeline_data)


def plot_tpr_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves a grouped bar chart comparing the True Positive Rate (TPR)
    of all pipelines across malicious categories.
    """
    malicious_categories = [
        'emotional_manipulation', 'encoding_attack', 'hypothetical_scenario',
        'multi_step_manipulation', 'persona_jailbreak', 'roleplay_jailbreak'
    ]
    malicious_df = df[df['category'].isin(malicious_categories)]

    if malicious_df.empty:
        print("No malicious category data to plot for TPR.")
        return

    plt.figure()
    ax = sns.barplot(data=malicious_df, x='category', y='tpr', hue='pipeline', dodge=True)

    ax.set_title('Comparison of Malicious Prompt Detection Rate (TPR) Across Pipelines', fontsize=18, weight='bold',
                 pad=20)
    ax.set_ylabel('Detection Rate (Higher is Better)', fontsize=14)
    ax.set_xlabel('Malicious Attack Category', fontsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=12)

    # --- ANNOTATION FIX IS HERE ---
    # This loop now explicitly places a "0%" label for bars with zero height.
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0%}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    fontsize=9, color='black',
                    xytext=(0, 5),
                    textcoords='offset points')

    plt.legend(title='Pipeline', fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "comparison_plot_tpr.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Successfully saved TPR comparison plot to: {output_path}")


def plot_fpr_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves a grouped bar chart comparing the False Positive Rate (FPR)
    of all pipelines across benign categories.
    """
    benign_categories = ['mundane_benign', 'sophisticated_benign']
    benign_df = df[df['category'].isin(benign_categories)]

    if benign_df.empty:
        print("No benign category data to plot for FPR.")
        return

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=benign_df, x='category', y='fpr', hue='pipeline', dodge=True)

    ax.set_title('Comparison of Benign Prompt False Positive Rate (FPR) Across Pipelines', fontsize=18, weight='bold',
                 pad=20)
    ax.set_ylabel('False Positive Rate (Lower is Better)', fontsize=14)
    ax.set_xlabel('Benign Prompt Category', fontsize=14)
    ax.tick_params(axis='x', rotation=0, labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1.05)

    # --- ANNOTATION FIX IS HERE ---
    # Same logic applied to the FPR chart for consistency.
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0%}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    fontsize=11, color='black',
                    xytext=(0, 5),
                    textcoords='offset points')

    plt.legend(title='Pipeline', fontsize=12)
    plt.tight_layout()

    output_path = output_dir / "comparison_plot_fpr.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Successfully saved FPR comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from multiple AI pipeline analysis reports.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-d', '--reports-dir',
        default='analysis_reports',
        help="Directory containing the analysis report subdirectories (e.g., 'analysis_svm_...')."
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='comparison_charts',
        help="Directory where the final comparison charts will be saved."
    )
    args = parser.parse_args()

    reports_path = Path(args.reports_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        combined_df = load_and_process_reports(reports_path)
        plot_tpr_comparison(combined_df, output_path)
        plot_fpr_comparison(combined_df, output_path)
        print("\nComparison chart generation complete.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()