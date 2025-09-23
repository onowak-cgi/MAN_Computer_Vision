#!/usr/bin/env python3
"""
Results Analyzer for Heat Protection Sleeve Classification
=========================================================

This script analyzes batch test results, creates confusion matrices,
and generates performance visualizations.

Usage:
    python results_analyzer.py [--results-file RESULTS.json] [--output-dir OUTPUT]
"""

import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import datetime

class ResultsAnalyzer:
    def __init__(self, output_dir="results"):
        """Initialize results analyzer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load results from JSON file"""
        results_path = Path(results_file)
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Loaded results from: {results_file}")
        print(f"📸 Total images: {results['metadata']['total_images']}")
        
        return results
    
    def prepare_data(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""
        
        # Extract result records
        records = []
        for result in results["results"]:
            record = {
                "image_name": result["image_name"],
                "ground_truth": result["ground_truth"],
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "evidence_summary": result["evidence_summary"],
                "correct": result["ground_truth"] == result["predicted_class"] if result["ground_truth"] != "Unknown" else None,
                "best_guess": result.get("best_guess"),
                "best_guess_confidence": result.get("best_guess_confidence"),
                "why_uncertain_reasons": (result.get("why_uncertain", {}) or {}).get("reasons", []),
                "why_uncertain_narrative": (result.get("why_uncertain", {}) or {}).get("narrative"),
                "missing_information_questions": result.get("missing_information_questions", []),
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Filter out errors and unknown ground truth for accuracy calculations
        df_clean = df[
            (df["predicted_class"] != "ERROR") & 
            (df["ground_truth"] != "Unknown")
        ].copy()
        
        print(f"📊 Clean data: {len(df_clean)} images with known ground truth")
        
        return df, df_clean
    
    def create_confusion_matrix(self, df_clean: pd.DataFrame, save_path: str = None) -> np.ndarray:
        """Create and visualize confusion matrix"""
        
        if len(df_clean) == 0:
            print("⚠️  No clean data available for confusion matrix")
            return np.array([])
        
        # Get unique labels
        all_labels = sorted(set(df_clean["ground_truth"].unique()) | set(df_clean["predicted_class"].unique()))
        
        # Create confusion matrix
        cm = confusion_matrix(
            df_clean["ground_truth"], 
            df_clean["predicted_class"],
            labels=all_labels
        )
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=all_labels,
            yticklabels=all_labels,
            cbar_kws={'label': 'Number of Images'}
        )
        
        plt.title('Confusion Matrix - Heat Protection Sleeve Classification', fontsize=16, pad=20)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        
        # Add accuracy information
        accuracy = accuracy_score(df_clean["ground_truth"], df_clean["predicted_class"])
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.2%}', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrix to: {save_path}")
        
        plt.show()
        
        return cm

    def create_confusion_matrix_with_best_guess(self, df: pd.DataFrame, save_path: str = None):
        """
        Create a confusion matrix where 'Uncertain' predictions are replaced by 'best_guess'
        (when available). Only uses rows with known ground truth and non-ERROR predictions.
        """
        df_bg = df[
            (df["ground_truth"] != "Unknown") &
            (df["predicted_class"] != "ERROR")
        ].copy()

        mask_uncertain = df_bg["predicted_class"] == "Uncertain"
        has_best_guess = mask_uncertain & df_bg["best_guess"].notna()
        df_bg.loc[has_best_guess, "predicted_class"] = df_bg.loc[has_best_guess, "best_guess"]

        return self.create_confusion_matrix(
            df_bg,
            save_path=save_path or (self.output_dir / "confusion_matrix_with_best_guess.png")
        )


    def calculate_metrics(self, df_clean: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        
        if len(df_clean) == 0:
            return {"error": "No clean data available"}
        
        y_true = df_clean["ground_truth"]
        y_pred = df_clean["predicted_class"]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=sorted(y_true.unique())
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Organize results
        labels = sorted(y_true.unique())
        per_class_metrics = {}
        
        for i, label in enumerate(labels):
            per_class_metrics[label] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": support[i]
            }
        
        metrics = {
            "overall": {
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1_weighted": f1_weighted,
                "total_samples": len(df_clean)
            },
            "per_class": per_class_metrics
        }
        
        return metrics
    
    def calculate_metrics_with_best_guess(self, df: pd.DataFrame) -> Dict[str, Any]:
        df_bg = df[
            (df["ground_truth"] != "Unknown") &
            (df["predicted_class"] != "ERROR")
        ].copy()
        uncertain_mask = df_bg["predicted_class"] == "Uncertain"
        with_best = uncertain_mask & df_bg["best_guess"].notna()
        df_bg.loc[with_best, "predicted_class"] = df_bg.loc[with_best, "best_guess"]
        return self.calculate_metrics(df_bg)


    def create_confidence_analysis(self, df: pd.DataFrame, save_path: str = None):
        """Analyze confidence scores"""
        
        # Filter out errors
        df_valid = df[df["predicted_class"] != "ERROR"].copy()
        
        if len(df_valid) == 0:
            print("⚠️  No valid predictions for confidence analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confidence distribution by predicted class
        axes[0, 0].hist([
            df_valid[df_valid["predicted_class"] == cls]["confidence"] 
            for cls in df_valid["predicted_class"].unique()
        ], 
        label=df_valid["predicted_class"].unique(),
        alpha=0.7, bins=20)
        axes[0, 0].set_title('Confidence Distribution by Predicted Class')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Confidence vs Correctness (if ground truth available)
        df_with_gt = df_valid[df_valid["ground_truth"] != "Unknown"].copy()
        if len(df_with_gt) > 0:
            correct_conf = df_with_gt[df_with_gt["correct"] == True]["confidence"]
            incorrect_conf = df_with_gt[df_with_gt["correct"] == False]["confidence"]
            
            axes[0, 1].boxplot([correct_conf, incorrect_conf], 
                              labels=['Correct', 'Incorrect'])
            axes[0, 1].set_title('Confidence Scores: Correct vs Incorrect Predictions')
            axes[0, 1].set_ylabel('Confidence Score')
        
        # 3. Confidence by predicted class (box plot)
        df_valid.boxplot(column='confidence', by='predicted_class', ax=axes[1, 0])
        axes[1, 0].set_title('Confidence Distribution by Predicted Class')
        axes[1, 0].set_xlabel('Predicted Class')
        axes[1, 0].set_ylabel('Confidence Score')
        
        # 4. Confidence threshold analysis
        if len(df_with_gt) > 0:
            thresholds = np.arange(0.1, 1.0, 0.05)
            accuracies = []
            
            for threshold in thresholds:
                high_conf = df_with_gt[df_with_gt["confidence"] >= threshold]
                if len(high_conf) > 0:
                    acc = (high_conf["correct"].sum() / len(high_conf))
                    accuracies.append(acc)
                else:
                    accuracies.append(0)
            
            axes[1, 1].plot(thresholds, accuracies, marker='o')
            axes[1, 1].set_title('Accuracy vs Confidence Threshold')
            axes[1, 1].set_xlabel('Confidence Threshold')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "confidence_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confidence analysis to: {save_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict[str, Any], metrics: Dict[str, Any],
                        metrics_bg: Dict[str, Any] = None, save_path: str = None) -> str:
        """Generate an HTML performance report"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Handle both old 'directory' and new 'directories' formats
        metadata = results.get("metadata", {}) or {}
        if "directories" in metadata and isinstance(metadata["directories"], list):
            test_dirs = ", ".join(metadata["directories"])
        else:
            test_dirs = metadata.get("directory", "Unknown")

        total_images_meta = metadata.get("total_images", "Unknown")

        # Basic HTML skeleton and styles
        html_content = f"""<!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Heat Protection Sleeve Classification Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .section {{ margin: 30px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .good {{ color: green; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .error {{ color: red; font-weight: bold; }}
        .small {{ color: #666; font-size: 12px; }}
        code, pre {{ background: #fafafa; border: 1px solid #eee; padding: 2px 4px; border-radius: 3px; }}
      </style>
    </head>
    <body>

      <div class="header">
        <h1>🔧 Heat Protection Sleeve Classification Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Test Directory:</strong> {test_dirs}</p>
        <p><strong>Total Images (metadata):</strong> {total_images_meta}</p>
      </div>
    """

        # Overall performance block
        if "error" not in metrics:
            overall = metrics.get("overall", {}) or {}

            def badge_class(acc: float) -> str:
                if acc is None:
                    return ""
                if acc > 0.8:
                    return "good"
                if acc > 0.6:
                    return "warning"
                return "error"

            acc = overall.get("accuracy", 0.0)
            html_content += f"""
      <div class="section">
        <h2>📊 Overall Performance</h2>
        <div class="metric">Accuracy: <span class="{badge_class(acc)}">{acc:.2%}</span></div>
        <div class="metric">Precision (Macro): {overall.get('precision_macro', 0.0):.3f}</div>
        <div class="metric">Recall (Macro): {overall.get('recall_macro', 0.0):.3f}</div>
        <div class="metric">F1-Score (Macro): {overall.get('f1_macro', 0.0):.3f}</div>
        <div class="metric">Total Samples (valid): {overall.get('total_samples', 0)}</div>
      </div>

      <div class="section">
        <h2>📈 Per-Class Performance</h2>
        <table>
          <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Support</th>
          </tr>
    """
            for class_name, class_metrics in (metrics.get("per_class", {}) or {}).items():
                html_content += f"""      <tr>
            <td>{class_name}</td>
            <td>{class_metrics.get('precision', 0.0):.3f}</td>
            <td>{class_metrics.get('recall', 0.0):.3f}</td>
            <td>{class_metrics.get('f1_score', 0.0):.3f}</td>
            <td>{class_metrics.get('support', 0)}</td>
          </tr>
    """
            html_content += "    </table>\n  </div>\n"

        # Uncertainty Analysis section (using per-image `results["results"]`)
        uncertain = [r for r in results.get("results", []) if r.get("predicted_class") == "Uncertain"]
        if uncertain:
            from collections import Counter
            reason_counter = Counter()
            question_counter = Counter()

            for r in uncertain:
                reasons = ((r.get("why_uncertain", {}) or {}).get("reasons", []) or [])
                reason_counter.update(reasons)
                qs = (r.get("missing_information_questions", []) or [])
                question_counter.update(qs)

            html_content += f"""
      <div class="section">
        <h2>🤔 Uncertainty Analysis</h2>
        <p>Total Uncertain: <strong>{len(uncertain)}</strong></p>
    """

            if reason_counter:
                html_content += "    <h3>Top Reasons</h3>\n    <ul>\n"
                for reason, cnt in reason_counter.most_common(10):
                    html_content += f"      <li>{reason}: {cnt}</li>\n"
                html_content += "    </ul>\n"

            if question_counter:
                html_content += "    <h3>Frequently Raised Questions (from model)</h3>\n    <ol>\n"
                for q, cnt in question_counter.most_common(10):
                    html_content += f"      <li>{q} <em>({cnt}×)</em></li>\n"
                html_content += "    </ol>\n"

            html_content += "  </div>\n"

        if metrics_bg and ("overall" in metrics_bg):
            overall_bg = metrics_bg.get("overall", {}) or {}
            html_content += f"""
      <div class="section">
        <h2>📊 Performance Using Best Guess (for Uncertain)</h2>
        <div class="metric">Accuracy (with best guess): <strong>{overall_bg.get('accuracy', 0.0):.2%}</strong></div>
        <div class="metric">Macro F1 (with best guess): {overall_bg.get('f1_macro', 0.0):.3f}</div>
        <p class="small"><em>Note:</em> This analysis view replaces 'Uncertain' predictions with their 'best_guess' solely for evaluation. Original stored predictions remain unchanged.</p>
      </div>
    """

        # Save report
        if save_path is None:
            save_path = self.output_dir / "performance_report.html"

        save_path = Path(save_path)
        save_path.write_text(html_content, encoding="utf-8")
        print(f"📄 Generated performance report: {save_path}")

        return str(save_path)

    
    def analyze_results(self, results_file: str) -> Dict[str, Any]:
        """Main analysis workflow"""
        print("🔍 Starting results analysis...")

        # 1) Load results
        results = self.load_results(results_file)

        # 2) Prepare data
        df, df_clean = self.prepare_data(results)

        # 3) Strict confusion matrix (saved to file)
        cm = self.create_confusion_matrix(
            df_clean,
            save_path=self.output_dir / "confusion_matrix.png"
        )

        # 4) Strict metrics
        metrics = self.calculate_metrics(df_clean)

        # 5) Confidence analysis (saved to file)
        #    Uses the full df (includes 'Unknown' GT filtering inside the plot logic where needed)
        self.create_confidence_analysis(
            df,
            save_path=self.output_dir / "confidence_analysis.png"
        )

        # 6) Best-guess view (optional but recommended)
        #    Replace 'Uncertain' predictions with their 'best_guess' for analysis only.
        cm_bg = []
        metrics_bg = None

        # These helpers should be added to ResultsAnalyzer as suggested earlier.
        # Guarded calls allow backward compatibility if they don't exist yet.
        if hasattr(self, "create_confusion_matrix_with_best_guess"):
            try:
                cm_bg_arr = self.create_confusion_matrix_with_best_guess(
                    df,
                    save_path=self.output_dir / "confusion_matrix_with_best_guess.png"
                )
                if cm_bg_arr is not None and getattr(cm_bg_arr, "size", 0) > 0:
                    cm_bg = cm_bg_arr.tolist()
            except Exception as e:
                print(f"⚠️  Skipped best-guess confusion matrix due to error: {e}")

        if hasattr(self, "calculate_metrics_with_best_guess"):
            try:
                metrics_bg = self.calculate_metrics_with_best_guess(df)
            except Exception as e:
                print(f"⚠️  Skipped best-guess metrics due to error: {e}")

        # 7) Generate HTML report (includes uncertainty analysis and optional best-guess section)
        # 🔧 If you don't want to include best-guess metrics in the report, pass `metrics_bg=None`.
        report_path = self.generate_report(results, metrics, metrics_bg=metrics_bg)

        # 8) Console summaries
        self.print_analysis_summary(metrics, total_images=len(df))
        self.print_binary_analysis(df_clean)

        print("\n✨ Analysis completed successfully!")
        print("📄 Check the performance report for detailed results")

        return {
            "metrics": metrics,
            "metrics_with_best_guess": metrics_bg,
            "confusion_matrix": cm.tolist() if getattr(cm, "size", 0) > 0 else [],
            "confusion_matrix_with_best_guess": cm_bg,
            "report_path": report_path,
        }

    
    def print_analysis_summary(self, metrics: Dict[str, Any], total_images: int):
        """Print analysis summary to console"""
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        if "error" in metrics:
            print(f"❌ {metrics['error']}")
            return
        
        overall = metrics["overall"]
        
        print(f"📸 Total Images Analyzed: {total_images}")
        print(f"✅ Valid Predictions: {overall['total_samples']}")
        print(f"🎯 Overall Accuracy: {overall['accuracy']:.2%}")
        print(f"📈 Macro F1-Score: {overall['f1_macro']:.3f}")
        
        print(f"\n🏷️  PER-CLASS PERFORMANCE:")
        for class_name, class_metrics in metrics["per_class"].items():
            print(f"   {class_name}:")
            print(f"      Precision: {class_metrics['precision']:.3f}")
            print(f"      Recall: {class_metrics['recall']:.3f}")
            print(f"      F1-Score: {class_metrics['f1_score']:.3f}")
            print(f"      Support: {class_metrics['support']}")
        
        # Uncertainty section
        uncertain = [r for r in results["results"] if r.get("predicted_class") == "Uncertain"]
        if uncertain:
            from collections import Counter
            reason_counter = Counter()
            question_counter = Counter()
            for r in uncertain:
                reason_counter.update((r.get("why_uncertain", {}) or {}).get("reasons", []) or [])
                question_counter.update(r.get("missing_information_questions", []) or [])

            html_content += """
            <div class="section">
              <h2>🤔 Uncertainty Analysis</h2>
              <p>Total Uncertain: <strong>{count}</strong></p>
            """.format(count=len(uncertain))

            if reason_counter:
                html_content += "<h3>Top Reasons</h3><ul>"
                for reason, cnt in reason_counter.most_common(10):
                    html_content += f"<li>{reason}: {cnt}</li>"
                html_content += "</ul>"

            if question_counter:
                html_content += "<h3>Frequently Raised Questions (from model)</h3><ol>"
                for q, cnt in question_counter.most_common(10):
                    html_content += f"<li>{q} <em>({cnt}×)</em></li>"
                html_content += "</ol>"

            html_content += "</div>"


    def print_binary_analysis(self, df_clean: pd.DataFrame):
        """Print a classification report for binary (OK/Broken) cases only."""
        
        print("\n" + "-" * 60)
        print("BINARY PERFORMANCE (excluding 'Uncertain' predictions)")
        print("-" * 60)
        
        df_binary = df_clean[df_clean["predicted_class"] != "Uncertain"].copy()
        
        if len(df_binary) == 0:
            print("⚠️ No predictions left after excluding 'Uncertain'. Cannot generate binary report.")
            return
            
        y_true = df_binary["ground_truth"]
        y_pred = df_binary["predicted_class"]
        
        # Ensure we only have 'OK' and 'Broken' in the true labels as well for a clean report
        df_binary_strict = df_binary[df_binary["ground_truth"].isin(["OK", "Broken"])]
        y_true_strict = df_binary_strict["ground_truth"]
        y_pred_strict = df_binary_strict["predicted_class"]

        if len(y_true_strict) > 0:
            report = classification_report(y_true_strict, y_pred_strict, labels=["OK", "Broken"])
            print(report)
        else:
            print("⚠️ No 'OK' or 'Broken' ground truth samples in the binary subset.")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Analyze heat protection sleeve classification results")
    parser.add_argument("--results-file", "-r",
                       default="results/latest_results.json",
                       help="JSON file containing batch test results")
    parser.add_argument("--output-dir", "-o",
                       default="results",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    print("📊 Heat Protection Sleeve Results Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    try:
        analyzer = ResultsAnalyzer(output_dir=args.output_dir)
        print("✅ Results analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return 1
    
    # Run analysis
    try:
        analysis_results = analyzer.analyze_results(args.results_file)
        
        print("\n✨ Analysis completed successfully!")
        print(f"📄 Check the performance report for detailed results")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
