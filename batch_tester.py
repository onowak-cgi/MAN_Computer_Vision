#!/usr/bin/env python3
"""
Batch Tester for Heat Protection Sleeve Classification
=====================================================

This script runs the heat protection sleeve classifier on multiple images,
saves results, and prepares data for analysis.

Usage:
    python batch_tester.py [--directory DIR] [--output OUTPUT_DIR]
"""

import os
import json
import csv
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from heat_protection_classifier import HeatProtectionClassifier

ENGINE_PATTERN = re.compile(r"^(?P<engine_id>\d+)_image_(?P<idx>\d+)\.(jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)

def parse_engine_id(filename: str) -> Tuple[str, int]:
    """
    Extract (engine_id, idx) from '147310880_image_4.jpeg'.
    """
    name = Path(filename).name
    m = ENGINE_PATTERN.match(name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    return m.group("engine_id"), int(m.group("idx"))

def group_images_by_engine(directories: List[str]) -> Tuple[Dict[str, List[Path]], Dict[str, str]]:
    """
    Group images by engine_id across given directories.
    Also infer ground truth label per engine from directory name:
      - 'correct' -> OK
      - 'notcorrect' -> Broken
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    gt_by_engine: Dict[str, str] = {}

    for d in directories:
        dpath = Path(d)
        if not dpath.exists():
            print(f"⚠️  Directory not found: {dpath}")
            continue
        dname = dpath.name.lower()
        if "correct" in dname and "not" not in dname:
            label = "OK"
        elif "notcorrect" in dname or "broken" in dname:
            label = "Broken"
        else:
            label = "Unknown"

        for p in dpath.glob("*.*"):
            if not p.is_file():
                continue
            try:
                engine_id, _ = parse_engine_id(p.name)
            except ValueError:
                continue
            groups[engine_id].append(p)
            prev = gt_by_engine.get(engine_id)
            if prev is None:
                gt_by_engine[engine_id] = label
            elif prev != label:
                gt_by_engine[engine_id] = "Unknown"  # conflict safeguard

    # Sort views per engine by index
    for engine_id, paths in groups.items():
        groups[engine_id] = sorted(paths, key=lambda p: parse_engine_id(p.name)[1])

    return groups, gt_by_engine

class BatchTester:
    def __init__(self, output_dir="results"):
        """Initialize batch tester with output directory"""
        self.classifier = HeatProtectionClassifier()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this test run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory"""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def parse_classification_result(self, result_text: str) -> Dict[str, Any]:
        """Parse the JSON result from the classifier"""
        try:
            # Extract JSON from the response (remove markdown formatting if present)
            if "```json" in result_text:
                start = result_text.find("```json") + 7
                end = result_text.find("```", start)
                json_text = result_text[start:end].strip()
            else:
                json_text = result_text.strip()
            
            return json.loads(json_text)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Failed to parse JSON result: {e}")
            return {
                "classification": "ERROR",
                "confidence": 0.0,
                "evidence_summary": f"JSON parsing error: {e}",
                "error": str(e)
            }
    
    def determine_ground_truth(self, image_path: str) -> str:
        """Determine ground truth label based on directory structure"""
        path = Path(image_path)
        
        if "notcorrect" in str(path.parent).lower():
            return "Broken"
        elif "correct" in str(path.parent).lower():
            return "OK"
        else:
            return "Unknown"
    
    def run_batch_test(self,
                       image_directories: List[str],
                       training_directories: List[str] = None) -> Dict[str, Any]:
        """
        Run engine-level batch classification using all views per engine and binary-only output.

        - Groups images by engine/case id using filename pattern: <engine_id>_image_<idx>.<ext>
        - Uses multi-view few-shot training examples from correct_training / notcorrect_training
        - Calls the Gemini classifier once per ENGINE, providing all its views jointly
        - Saves ENGINE-level results (one row per engine) and prints a binary-only summary

        Parameters
        ----------
        image_directories : List[str]
            Test directories (e.g., ["Archiv/correct", "Archiv/notcorrect"])
        training_directories : List[str], optional
            Training directories (e.g., ["Archiv/correct_training", "Archiv/notcorrect_training"])

        Returns
        -------
        Dict[str, Any]
            Batch results containing metadata and engine-level results.
        """
        import time
        from pathlib import Path

        # Lazy import to avoid circular dependencies if any
        try:
            from heat_protection_classifier import HeatProtectionClassifier
        except Exception as e:
            raise ImportError(f"Failed to import HeatProtectionClassifier: {e}")

        print("🚀 Starting engine-level batch test (multi-view, binary-only)")
        t0 = time.time()

        # Default training directories if not provided
        if not training_directories or len(training_directories) < 2:
            training_directories = ["Archiv/correct_training", "Archiv/notcorrect_training"]

        # Instantiate the Gemini-based classifier
        classifier = HeatProtectionClassifier()

        # Perform engine-level classification using all views and multi-view training exemplars
        try:
            engine_level_results = classifier.classify_grouped_engines(
                test_dirs=image_directories,
                train_correct_dir=training_directories[0],
                train_notcorrect_dir=training_directories[1],
            )
        except Exception as e:
            print(f"❌ Classification failed at engine-batch level: {e}")
            # Fail-safe empty results so we still produce a metadata shell
            engine_level_results = []

        # Ensure required fields exist for downstream CSV flattening
        # (Some fields like processing_time may not be provided by the classifier)
        for r in engine_level_results:
            r.setdefault("processing_time", 0.0)  # keep schema stable
            r.setdefault("confidence", None)
            r.setdefault("evidence_summary", "")

        # Compute metadata
        total_engines = len(engine_level_results)
        total_images = sum(len(r.get("image_names", [])) for r in engine_level_results)

        batch_results = {
            "metadata": {
                "directories": image_directories,
                "training_directories": training_directories,
                "total_images": total_images,
                "total_engines": total_engines,
                "timestamp": self.timestamp,
            },
            "results": engine_level_results,  # ENGINE-LEVEL records
        }

        # Save and summarize
        try:
            self.save_results(batch_results)
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")

        try:
            self.print_summary(batch_results)  # See B5 notes below for binary-only/engine wording
        except Exception as e:
            print(f"⚠️  Failed to print summary: {e}")

        print(f"⏱️ Total elapsed: {time.time() - t0:.2f}s")
        return batch_results

    def save_results(self, batch_results: Dict[str, Any]):
        """Save results in multiple formats"""
        
        # Save raw JSON results
        json_file = self.output_dir / f"raw_results_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"💾 Saved raw results to: {json_file}")
        
        # Save processed CSV results
        csv_file = self.output_dir / f"processed_results_{self.timestamp}.csv"
        
        # Flatten results for CSV
        csv_data = []
        for result in batch_results["results"]:
            csv_row = {
                "engine_id": result.get("engine_id"),
                "image_names": ";".join(result.get("image_names", [])),
                "image_count": len(result.get("image_names", [])),
                "image_paths": ";".join(result.get("image_paths", [])),
                "ground_truth": result.get("ground_truth"),
                "predicted_class": result.get("predicted_class"),
                "confidence": result.get("confidence"),
                "evidence_summary": result.get("evidence_summary"),
                "processing_time": result.get("processing_time"),
            }

            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved processed results to: {csv_file}")
        
        # Save latest results (for easy access)
        latest_json = self.output_dir / "latest_results.json"
        latest_csv = self.output_dir / "latest_results.csv"
        
        with open(latest_json, 'w') as f:
            json.dump(batch_results, f, indent=2)
        df.to_csv(latest_csv, index=False)
        
        print(f"💾 Saved latest results to: {latest_json} and {latest_csv}")
    
    def print_summary(self, batch_results: Dict[str, Any]):
        """Print a summary of the batch test results (ENGINE-LEVEL, binary-only)"""
        results = batch_results["results"]
        total = len(results)  # B5: engines, not images
    
        if total == 0:
            print("❌ No results to summarize")
            return
    
        # Count predictions
        predictions = {}
        errors = 0
        correct_predictions = 0
        total_with_ground_truth = 0
        successful_classifications = 0
    
        # B5: No 'Uncertain' handling — binary only: OK/Broken (+ ERROR for failures)
        for result in results:
            pred_class = result.get("predicted_class", "ERROR")
            # B5: Normalize unexpected labels to ERROR (safety against legacy runs)
            if pred_class not in {"OK", "Broken", "ERROR"}:
                pred_class = "ERROR"
    
            predictions[pred_class] = predictions.get(pred_class, 0) + 1
    
            if pred_class == "ERROR":
                errors += 1
            else:
                # Only count non-ERROR cases for accuracy calculation
                gt = result.get("ground_truth", "Unknown")
                if gt != "Unknown":
                    successful_classifications += 1
                    if pred_class == gt:
                        correct_predictions += 1
    
            # Count total with ground truth (including errors for reporting)
            if result.get("ground_truth", "Unknown") != "Unknown":
                total_with_ground_truth += 1
    
        print("\n" + "=" * 60)
        print("📊 BATCH TEST SUMMARY (ENGINE-LEVEL)")  # B5: clarify engine-level
        print("=" * 60)
        print(f"🧩 Total engines processed: {total}")    # B5: engines wording
        print(f"❌ Errors: {errors}")
        print(f"✅ Successful classifications: {total - errors}")
    
        print(f"\n🎯 PREDICTION DISTRIBUTION:")
        for pred_class, count in sorted(predictions.items()):
            percentage = (count / total) * 100 if total > 0 else 0.0
            print(f"   {pred_class}: {count} ({percentage:.1f}%)")
    
        # B5: Accuracy is binary-only (OK/Broken), excludes API failures (ERROR)
        if successful_classifications > 0:
            accuracy = (correct_predictions / successful_classifications) * 100
            print(f"\n🎯 ACCURACY (excluding API failures): "
                  f"{correct_predictions}/{successful_classifications} ({accuracy:.1f}%)")
            if errors > 0:
                print(f"   📝 Note: {errors} API failures excluded from accuracy calculation")
        elif total_with_ground_truth > 0:
            print(f"\n⚠️  All {total_with_ground_truth} engines with ground truth had API failures "
                  f"- no accuracy can be calculated")
    
        # Average confidence (exclude ERROR)
        confidences = [r.get("confidence") for r in results
                       if r.get("predicted_class") != "ERROR" and r.get("confidence") is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"📈 Average confidence: {avg_confidence:.2f}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Batch test heat protection sleeve classifier on multiple directories.")
    parser.add_argument("--directories", "-d", 
                       nargs='+',
                       default=["Archiv/correct", "Archiv/notcorrect"],
                       help="One or more directories containing images to test.")
    parser.add_argument("--output", "-o", 
                       default="results",
                       help="Output directory for results")
    parser.add_argument("--train-correct", default="Archiv/correct_training",
                        help="Directory with OK training images (multi-view engines).")
    parser.add_argument("--train-notcorrect", default="Archiv/notcorrect_training",
                        help="Directory with Broken training images (multi-view engines).")

    
    args = parser.parse_args()
    
    print("🔧 Heat Protection Sleeve Batch Tester")
    print("=" * 50)
    
    # Initialize tester
    try:
        tester = BatchTester(output_dir=args.output)
        print("✅ Batch tester initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return 1
    
    # Run batch test
    try:
        results = tester.run_batch_test(
            image_directories=args.directories,
            training_directories=[args.train_correct, args.train_notcorrect]
        )

        
        print("\n✨ Batch testing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Batch testing failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
