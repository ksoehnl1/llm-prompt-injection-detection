# evaluation_analyzer.py
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Plotting Style Configuration ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

''' 
NOTE: This was made with the HELP of ChatGPT - it is nothing more than a metrics and plot creator.

Creates simple reports, plots, and metrics for use.
'''

class HumanEvaluationAnalyzer:
    """
    Analyzes and visualizes human-annotated pipeline results to generate
    comprehensive, shareable reports.
    """

    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def load_human_evaluations(self, evaluation_file: Path) -> Optional[pd.DataFrame]:
        """Loads and validates the annotated CSV file."""
        try:
            df = pd.read_csv(evaluation_file)
            required = ['prompt_id', 'original_label', 'human_verdict']
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV is missing one of the required columns: {required}")
            return df
        except FileNotFoundError:
            print(f"Error: Evaluation file not found at {evaluation_file}")
            return None
        except Exception as e:
            print(f"Error reading or validating CSV file: {e}")
            return None

    def _compute_rates_from_df(self, df: pd.DataFrame) -> Dict:
        """
        Core logic to calculate key metrics from a DataFrame of verdicts.
        This function is applied to the whole dataset and to each category group.
        """
        malicious_df = df[df['original_label'] == 'malicious']
        benign_df = df[df['original_label'] == 'benign']

        # --- Malicious Prompt Analysis (TPR, FNR) ---
        malicious_total = len(malicious_df)
        tpr, fnr, fnr_harmful, fnr_benign_outcome = 0.0, 0.0, 0.0, 0.0
        if malicious_total > 0:
            counts = malicious_df['human_verdict'].value_counts()
            correct_blocks = counts.get('CORRECT_BLOCK', 0)
            fn_harmful = counts.get('FALSE_NEGATIVE_HARMFUL', 0)
            fn_benign_outcome = counts.get('FALSE_NEGATIVE_BENIGN_OUTCOME', 0)

            tpr = correct_blocks / malicious_total
            fnr = (fn_harmful + fn_benign_outcome) / malicious_total
            fnr_harmful = fn_harmful / malicious_total
            fnr_benign_outcome = fn_benign_outcome / malicious_total

        # --- Benign Prompt Analysis (TNR, FPR) ---
        benign_total = len(benign_df)
        tnr, fpr = 0.0, 0.0
        if benign_total > 0:
            counts = benign_df['human_verdict'].value_counts()
            correct_passes = counts.get('CORRECT_PASS', 0)
            false_positives = counts.get('FALSE_POSITIVE', 0)

            tnr = correct_passes / benign_total
            fpr = false_positives / benign_total

        return {
            "true_positive_rate": tpr,
            "false_negative_rate": fnr,
            "false_negative_rate_harmful": fnr_harmful,
            "false_negative_rate_benign_outcome": fnr_benign_outcome,
            "true_negative_rate": tnr,
            "false_positive_rate": fpr,
            "malicious_count": malicious_total,
            "benign_count": benign_total,
            "total_count": len(df)
        }

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """REMOVED 'difficulty': Calculates overall metrics and breakdowns by category."""
        evaluated_df = df[df['human_verdict'].notna()].copy()
        if evaluated_df.empty:
            return {"error": "No completed human evaluations found in the file."}

        overall_metrics = self._compute_rates_from_df(evaluated_df)

        by_category = evaluated_df.groupby('category').apply(self._compute_rates_from_df).to_dict()

        metrics = {
            "overall_performance": overall_metrics,
            "by_category": by_category,
        }

        if 'response_time' in df.columns:
            valid_times = df['response_time'].dropna()
            if not valid_times.empty:
                metrics['response_time_stats'] = {
                    "mean": valid_times.mean(), "median": valid_times.median(),
                    "min": valid_times.min(), "max": valid_times.max(),
                    "p95": valid_times.quantile(0.95)
                }
        return metrics

    def _generate_summary_text(self, metrics: Dict, filename: str) -> str:
        """REMOVED 'difficulty': Creates a text summary with category breakdown only."""
        overall = metrics.get('overall_performance', {})
        summary = [
            "==================================================",
            f"      HUMAN EVALUATION ANALYSIS REPORT",
            "==================================================",
            f"Source File: {filename}",
            f"Analysis Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"\n--- OVERVIEW ---",
            f"Total Prompts Evaluated: {overall.get('total_count', 0)}",
            f"  - Malicious Samples: {overall.get('malicious_count', 0)}",
            f"  - Benign Samples: {overall.get('benign_count', 0)}",
            "\n--- CORE SECURITY & UX METRICS ---",
            f"Malicious Prompt Detection Rate (TPR): {overall.get('true_positive_rate', 0):.1%}",
            f"Benign Prompt False Positive Rate (FPR): {overall.get('false_positive_rate', 0):.1%}",
            f"\n--- CRITICAL FAILURE ANALYSIS ---",
            f"Overall False Negative Rate (FNR): {overall.get('false_negative_rate', 0):.1%}",
            f"  - Breakdown: Harmful Outcome Rate:   {overall.get('false_negative_rate_harmful', 0):.1%}",
            f"  - Breakdown: Benign Outcome Rate:    {overall.get('false_negative_rate_benign_outcome', 0):.1%}",
        ]

        def format_breakdown(title: str, data: Dict):
            summary.append(f"\n--- {title.upper()} ---")
            if not data:
                summary.append("  (No data available)")
                return

            sorted_items = sorted(data.items(), key=lambda x: x[1].get('total_count', 0), reverse=True)

            for name, item in sorted_items:
                name_str = str(name) if pd.notna(name) else "Uncategorized"
                summary.append(f"  - {name_str} ({item.get('total_count', 0)} samples):")
                summary.append(
                    f"    TPR: {item.get('true_positive_rate', 0):.1%} | FPR: {item.get('false_positive_rate', 0):.1%} | FNR: {item.get('false_negative_rate', 0):.1%}")

        format_breakdown("PERFORMANCE BY CATEGORY", metrics.get('by_category', {}))
        # REMOVED: Call to format difficulty breakdown is gone.
        summary.append("\n==================================================")
        return "\n".join(summary)

    def _plot_tpr_breakdown(self, metrics: Dict, output_path: Path):
        """
        NEW: Plots ONLY the True Positive Rate (Detection Rate) for malicious categories.
        """
        data = metrics.get('by_category', {})
        if not data: return

        # Filter for malicious categories that have a TPR
        malicious_data = {
            cat: vals for cat, vals in data.items()
            if vals.get('malicious_count', 0) > 0
        }
        if not malicious_data: return

        df = pd.DataFrame.from_dict(malicious_data, orient='index').reset_index()
        df.rename(columns={'index': 'Category', 'true_positive_rate': 'Detection Rate (TPR)'}, inplace=True)

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=df, x='Category', y='Detection Rate (TPR)', color='cornflowerblue')

        plt.title('Malicious Prompt Detection Rate (TPR) by Attack Category', fontsize=18, weight='bold')
        plt.ylabel('Detection Rate', fontsize=12)
        plt.xlabel('Attack Category', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                        textcoords='offset points')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"      -> Saved TPR plot: {output_path.name}")

    def _plot_fpr_breakdown(self, metrics: Dict, output_path: Path):
        """
        NEW: Plots ONLY the False Positive Rate for benign categories.
        This correctly visualizes the 0% error rate for mundane prompts.
        """
        data = metrics.get('by_category', {})
        if not data: return

        # Filter for benign categories that have an FPR
        benign_data = {
            cat: vals for cat, vals in data.items()
            if vals.get('benign_count', 0) > 0
        }
        if not benign_data: return

        df = pd.DataFrame.from_dict(benign_data, orient='index').reset_index()
        df.rename(columns={'index': 'Category', 'false_positive_rate': 'False Positive Rate (FPR)'}, inplace=True)

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=df, x='Category', y='False Positive Rate (FPR)', color='sandybrown')

        plt.title('Benign Prompt False Positive Rate (FPR) by Category', fontsize=18, weight='bold')
        plt.ylabel('False Positive Rate', fontsize=12)
        plt.xlabel('Benign Category', fontsize=12)
        plt.xticks(rotation=0)
        plt.ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"      -> Saved FPR plot: {output_path.name}")

    def _plot_response_times(self, df: pd.DataFrame, output_path: Path):
        """Plots distribution of response times."""
        if 'response_time' not in df.columns or df['response_time'].dropna().empty: return
        plt.figure()
        sns.histplot(df['response_time'].dropna(), kde=True, bins=30)
        plt.title('Distribution of End-to-End Response Times', fontsize=18, weight='bold')
        plt.xlabel('Response Time (seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"      -> Saved plot: {output_path.name}")

    def generate_full_report(self, evaluation_file: str):
        eval_path = Path(evaluation_file)
        print(f"\n--- Generating Full Report for {eval_path.name} ---")
        df = self.load_human_evaluations(eval_path)
        if df is None: return

        report_dir_name = f"analysis_{eval_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
        report_dir = self.results_dir / report_dir_name
        report_dir.mkdir(exist_ok=True)
        print(f"  -> Saving report to: {report_dir}")

        metrics = self.calculate_metrics(df)
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return

        json_path = report_dir / "report_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"      -> Saved detailed metrics to: {json_path.name}")

        summary_text = self._generate_summary_text(metrics, eval_path.name)
        summary_path = report_dir / "summary_report.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"      -> Saved text summary to: {summary_path.name}")

        # REPLACED: Call the two new plotting functions instead of the old one.
        self._plot_tpr_breakdown(metrics, report_dir / 'plot_tpr_by_category.png')
        self._plot_fpr_breakdown(metrics, report_dir / 'plot_fpr_by_category.png')
        self._plot_response_times(df, report_dir / 'plot_response_times.png')

        print(f"--- Report Generation Complete ---")
        print(f"\nCONSOLE SUMMARY:\n{summary_text}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze human-annotated AI pipeline results.")
    parser.add_argument('-f', '--evaluation-file', required=True, help="Path to the annotated CSV file.")
    parser.add_argument('-d', '--results-dir', default='analysis_reports',
                        help="Directory where analysis reports will be saved.")
    args = parser.parse_args()

    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("\n--- Missing Required Libraries ---")
        print("This script requires 'matplotlib' and 'seaborn' for generating plots.")
        print("Please install them by running:\n\n    pip install matplotlib seaborn pandas\n")
        return

    analyzer = HumanEvaluationAnalyzer(results_dir=args.results_dir)
    analyzer.generate_full_report(evaluation_file=args.evaluation_file)


if __name__ == "__main__":
    main()