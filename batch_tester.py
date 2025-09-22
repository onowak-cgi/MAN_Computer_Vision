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

from heat_protection_classifier import HeatProtectionClassifier

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
    
    def run_batch_test(self, image_directories: List[str]) -> Dict[str, Any]:
        """
        Run batch classification on all images from multiple directories.

        Args:
            image_directories: A list of directories containing images to test.

        Returns:
            Dictionary containing all results and metadata.
        """
        print(f"🔄 Starting batch test on directories: {image_directories}")
        print(f"📁 Output directory: {self.output_dir}")

        # Get all image files from all specified directories
        all_image_files = []
        for directory in image_directories:
            try:
                all_image_files.extend(self.get_image_files(directory))
            except FileNotFoundError as e:
                print(f"⚠️  Warning: {e}")
        
        print(f"📸 Found a total of {len(all_image_files)} images to process.")

        if not all_image_files:
            print("❌ No images found in any of the specified directories!")
            return {}

        # Initialize results structure
        batch_results = {
            "metadata": {
                "timestamp": self.timestamp,
                "directories": [str(d) for d in image_directories],
                "total_images": len(all_image_files),
            },
            "results": []
        }
        
        # Process each image
        for i, image_path in enumerate(all_image_files, 1):
            print(f"\n--- Processing {i}/{len(all_image_files)}: {Path(image_path).name} ---")
            
            try:
                # Classify image
                raw_result = self.classifier.classify_image(image_path)
                parsed_result = self.parse_classification_result(raw_result)
                
                # Determine ground truth for the current image
                ground_truth = self.determine_ground_truth(image_path)

                # Create result record
                result_record = {
                    "image_path": str(image_path),
                    "image_name": Path(image_path).name,
                    "ground_truth": ground_truth,
                    "predicted_class": parsed_result.get("classification", "ERROR"),
                    "confidence": parsed_result.get("confidence", 0.0),
                    "evidence_summary": parsed_result.get("evidence_summary", ""),
                    "raw_response": raw_result,
                    "parsed_response": parsed_result,
                    "processing_time": datetime.datetime.now().isoformat()
                }
                
                batch_results["results"].append(result_record)
                
                # Print result
                print(f"✅ Result: {result_record['predicted_class']} (confidence: {result_record['confidence']:.2f})")
                if result_record["ground_truth"] != "Unknown":
                    correct = result_record['predicted_class'] == result_record["ground_truth"]
                    print(f"🎯 Ground Truth: {result_record['ground_truth']} - {'✅ CORRECT' if correct else '❌ INCORRECT'}")
                
            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                
                # Add error record
                error_record = {
                    "image_path": str(image_path),
                    "image_name": Path(image_path).name,
                    "ground_truth": self.determine_ground_truth(image_path),
                    "predicted_class": "ERROR",
                    "confidence": 0.0,
                    "evidence_summary": f"Processing error: {e}",
                    "raw_response": f"ERROR: {e}",
                    "parsed_response": {"error": str(e)},
                    "processing_time": datetime.datetime.now().isoformat()
                }
                
                batch_results["results"].append(error_record)
        
        # Save results
        self.save_results(batch_results)
        
        # Print summary
        self.print_summary(batch_results)
        
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
                "image_name": result["image_name"],
                "image_path": result["image_path"],
                "ground_truth": result["ground_truth"],
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "evidence_summary": result["evidence_summary"],
                "processing_time": result["processing_time"]
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
        """Print a summary of the batch test results"""
        results = batch_results["results"]
        total = len(results)
        
        if total == 0:
            print("❌ No results to summarize")
            return
        
        # Count predictions
        predictions = {}
        errors = 0
        correct_predictions = 0
        total_with_ground_truth = 0
        successful_classifications = 0
        
        for result in results:
            pred_class = result["predicted_class"]
            predictions[pred_class] = predictions.get(pred_class, 0) + 1
            
            if pred_class == "ERROR":
                errors += 1
            else:
                # Only count non-ERROR cases for accuracy calculation
                if result["ground_truth"] != "Unknown":
                    successful_classifications += 1
                    if result["predicted_class"] == result["ground_truth"]:
                        correct_predictions += 1
            
            # Count total with ground truth (including errors for reporting)
            if result["ground_truth"] != "Unknown":
                total_with_ground_truth += 1
        
        print("\n" + "=" * 60)
        print("📊 BATCH TEST SUMMARY")
        print("=" * 60)
        print(f"📸 Total images processed: {total}")
        print(f"❌ Errors: {errors}")
        print(f"✅ Successful classifications: {total - errors}")
        
        print(f"\n🎯 PREDICTION DISTRIBUTION:")
        for pred_class, count in sorted(predictions.items()):
            percentage = (count / total) * 100
            print(f"   {pred_class}: {count} ({percentage:.1f}%)")
        
        if successful_classifications > 0:
            accuracy = (correct_predictions / successful_classifications) * 100
            print(f"\n🎯 ACCURACY (excluding API failures): {correct_predictions}/{successful_classifications} ({accuracy:.1f}%)")
            if errors > 0:
                print(f"   📝 Note: {errors} API failures excluded from accuracy calculation")
        elif total_with_ground_truth > 0:
            print(f"\n⚠️  All {total_with_ground_truth} images with ground truth had API failures - no accuracy can be calculated")
        
        # Average confidence
        confidences = [r["confidence"] for r in results if r["predicted_class"] != "ERROR"]
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
        results = tester.run_batch_test(image_directories=args.directories)
        
        print("\n✨ Batch testing completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Batch testing failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
