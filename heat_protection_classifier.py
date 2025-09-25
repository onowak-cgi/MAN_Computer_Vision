#!/usr/bin/env python3
"""
Heat Protection Sleeve Classifier
=================================

This script uses Google Gemini Vision API with few-shot learning to classify
heat protection sleeves in automotive engine images as OK, Broken, or Uncertain.

Usage:
    python heat_protection_classifier.py [target_image_path]

If no target image is provided, it will use a default test image.
"""

import os
import sys
import json
import datetime
import logging
import tenacity
import mimetypes
from pathlib import Path
from google import genai
from google.genai import types
from google.api_core import exceptions
from dotenv import load_dotenv
# 🔧 FILE: heat_protection_classifier.py
# ADD after imports
import re
from collections import defaultdict
from typing import Dict, List, Tuple

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

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY_2")

# Configure logging to show tenacity's retry attempts
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# SYSTEM_PROMPT (binary only)
SYSTEM_PROMPT = """
Role & Objective
You are a senior automotive technician specialized in visual inspection of heat protection sleeves (HPS) in engine compartments. Your task is to classify each input engine as OK or Broken with a clear, evidence‑based rationale, focusing only on the heat protection sleeve and its immediate context. You may receive multiple images of the SAME engine (same case id); consider ALL views jointly to decide.

1) Scope & Definitions
Target component: Heat Protection Sleeve (HPS) — a protective sheath (often reflective foil, braided fiberglass, textile, or black sleeve) that shields hoses/wires from thermal sources (e.g., exhaust manifold, turbo, EGR pipes).
Hot zone proximity: Any area close to metallic parts that typically run hot (exhaust/turbo housings, EGR piping, DPF lines). Sleeves are expected primarily in these zones.
Underlying line: The hose/wire/conduit that the sleeve protects.

2) Decision Classes & Core Criteria
A. OK (Healthy) — All must be true (allowing minor, non-critical exceptions):
- Presence & Coverage: A sleeve is present where expected and coverage is continuous across the heat-critical segment.
- Integrity: No significant fraying/tears/holes/melted/charred areas or split seams exposing the line in the heat zone.
- Positioning: Not obviously slipped back; terminations intentional; fasteners serviceable.
- Cosmetic soiling (dust/dullness) is acceptable.
- Brief intentional exposure at connectors/bends/branches outside the heat-critical segment is acceptable.

B. Broken (Faulty)
- Missing/Displaced Sleeve in the heat-critical segment.
- Structural damage exposing the line in heat-critical area (fray/tear/hole/burn/melted/split seam).
- Inadequate coverage in heat-critical segment (e.g., >~2–3 cm gap).
- Fastener failure causing exposure in heat-critical area.

3) Output Format (Strict JSON)
Return only the following JSON (no extra commentary):

{
  "classification": "OK" | "Broken",
  "confidence": 0.0,
  "evidence_summary": "Succinct visual rationale tied to the heat zone and sleeve condition.",
  "observations": {
    "sleeve_presence": "present" | "absent" | "occluded",
    "coverage_in_heat_zone": "continuous" | "partial_gap" | "absent" | "uncertain",
    "integrity": ["no_damage", "fray", "tear", "hole", "burn_char", "melted", "split_seam", "unknown"],
    "positioning": ["well_positioned", "slipped_back", "loose_end", "missing_fastener", "unknown"],
    "hot_zone_cues": ["exhaust_metal_nearby", "turbo_housing", "egr_pipe", "none_visible", "occluded"],
    "cosmetics": ["dusty", "clean", "oily", "glare", "shadowed"]
  },
  "roi_notes": "Describe where the heat zone and sleeve were inspected (landmarks, relative positions).",
  "pitfall_checks": ["distinguish_cosmetic_vs_structural", "text_on_hose_not_conclusive", "angle_occlusion_checked", "component_mis-ID_checked"]
}

Validation rules:
- classification MUST be either "OK" or "Broken".
- Return only the JSON object described above.

Return only the JSON object (no Markdown, no triple backticks, no commentary).
"""

USER_PROMPT = "Classify this ENGINE using all provided views jointly. Focus on the hot-zone segment and return ONLY the strict JSON (OK or Broken)."

MAX_TRAIN_ENGINES_PER_CLASS = 3
MAX_IMAGES_PER_ENGINE_IN_PROMPT = 6

class HeatProtectionClassifier:

    def classify_grouped_engines(self,
                                 test_dirs: List[str],
                                 train_correct_dir: str = "Archiv/correct_training",
                                 train_notcorrect_dir: str = "Archiv/notcorrect_training"):
        """
        Classify all ENGINES found in the given test directories, using multi-view training exemplars.
        Returns a list of engine-level result dicts.
        """
        # Prepare training examples
        training_examples = self.select_training_examples_from_dirs(train_correct_dir, train_notcorrect_dir)

        # Group test engines
        test_groups, test_gt = group_images_by_engine(test_dirs)

        all_results = []
        for engine_id, paths in test_groups.items():
            try:
                raw_text = self.classify_engine_views(engine_id, paths, training_examples)
                try:
                    result_json = json.loads(raw_text)
                except json.JSONDecodeError:
                    result_json = {"classification": "ERROR", "confidence": 0.0, "evidence_summary": f"Non-JSON output: {raw_text[:200]}..."}

                all_results.append({
                    "engine_id": engine_id,
                    "image_names": [p.name for p in paths],
                    "image_paths": [str(p) for p in paths],
                    "ground_truth": test_gt.get(engine_id, "Unknown"),
                    "predicted_class": result_json.get("classification", "ERROR"),
                    "confidence": result_json.get("confidence"),
                    "evidence_summary": result_json.get("evidence_summary"),
                })
            except Exception as e:
                all_results.append({
                    "engine_id": engine_id,
                    "image_names": [p.name for p in paths],
                    "image_paths": [str(p) for p in paths],
                    "ground_truth": test_gt.get(engine_id, "Unknown"),
                    "predicted_class": "ERROR",
                    "confidence": 0.0,
                    "evidence_summary": f"Exception: {e}",
                })
        return all_results


    def classify_engine_views(self, engine_id: str, image_paths: List[Path], training_examples: Dict[str, List[Dict]]):
        """
        Classify a SINGLE ENGINE using ALL its views jointly.
        Returns the JSON text from Gemini.
        """
        # Upload engine views
        engine_files = self._upload_engine_views(image_paths)
        if not engine_files:
            raise FileNotFoundError(f"No valid views for engine {engine_id}")

        # Build contents
        contents = self._build_contents_for_engine(engine_id, engine_files, training_examples)

        # Generate with retry
        try:
            print(f"🤖 Classifying engine {engine_id} with {len(engine_files)} views ...")
            response = self._generate_content_with_retry(
                model="gemini-2.5-pro",
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            print("✅ Engine classification completed")
            return response.text
        except Exception as e:
            raise Exception(f"Engine classification failed: {e}")


    def _build_contents_for_engine(self, engine_id: str, engine_files: List, training_examples: Dict[str, List[Dict]]):
        """
        Construct the Gemini 'contents' sequence:
          - multiple training exemplars (OK/Broken), each with multiple images + label text
          - a TARGET section containing all views of the engine to classify + USER_PROMPT
        """
        contents = []

        # Few-shot multi-view training exemplars
        for label in ("OK", "Broken"):
            for ex in training_examples.get(label, []):
                contents.append(f"TRAINING EXAMPLE ({label}) — Engine {ex['engine_id']} — multiple views follow:")
                for f in ex["files"]:
                    contents.append(f)
                contents.append(f"Label: {label} (binary schema)")

        # Target engine with all views
        contents.append(f"TARGET ENGINE ({engine_id}) — multiple views follow:")
        for f in engine_files:
            contents.append(f)
        contents.append(USER_PROMPT)  # instruct to output only strict JSON

        return contents


    def _upload_engine_views(self, paths: List[Path]):
        """Upload multiple local paths and return file objects."""
        uploaded = []
        for p in paths[:MAX_IMAGES_PER_ENGINE_IN_PROMPT]:
            if not p.exists():
                print(f"⚠️ Missing file: {p}")
                continue
            try:
                fobj = self.client.files.upload(file=str(p))
                uploaded.append(fobj)
                print(f"✅ Uploaded: {p.name}")
            except Exception as e:
                print(f"❌ Upload failed for {p}: {e}")
        return uploaded

    def select_training_examples_from_dirs(self, correct_training_dir: str, notcorrect_training_dir: str):
        """
        Load multi-view training engines from the given directories, upload their images,
        and return dict with examples per class.
        """
        train_dirs = [correct_training_dir, notcorrect_training_dir]
        groups, gt = group_images_by_engine(train_dirs)

        ok_engines = [(eid, paths) for eid, paths in groups.items() if gt.get(eid) == "OK"]
        br_engines = [(eid, paths) for eid, paths in groups.items() if gt.get(eid) == "Broken"]

        ok_engines = ok_engines[:MAX_TRAIN_ENGINES_PER_CLASS]
        br_engines = br_engines[:MAX_TRAIN_ENGINES_PER_CLASS]

        examples = {"OK": [], "Broken": []}
        for label, engines in (("OK", ok_engines), ("Broken", br_engines)):
            for eid, paths in engines:
                uploaded_files = self._upload_engine_views(paths)
                if uploaded_files:
                    examples[label].append({"engine_id": eid, "files": uploaded_files})
        print(f"📦 Training examples -> OK: {len(examples['OK'])}, Broken: {len(examples['Broken'])}")
        return examples

    def __init__(self, api_key: str = None):
        """Initialize the classifier with Google Gemini client"""
        # Use the provided api_key or fall back to the one from the .env file
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("API_KEY_2 not found in .env file or not provided.")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Define example image paths
        self.pos_paths = [
            "Archiv/correct/147368364_image_2.jpeg",
            "Archiv/correct/147457017_image_2.jpeg",
            "Archiv/correct/147533541_image_1.jpeg",
        ]
        
        self.neg_paths = [
            "Archiv/notcorrect/145010971_image_1.jpeg",
            "Archiv/notcorrect/145322610_image_2.jpeg",
            "Archiv/notcorrect/145623226_image_2.jpeg",
        ]
        
        # Cache for uploaded files (valid for ~48 hours)
        self.cache_file = Path("example_files_cache.json")
        self._pos_files = None
        self._neg_files = None
    
    def _upload_many(self, paths):
        """Upload multiple images to Google's servers"""
        uploaded_files = []
        for path in paths:
            if not os.path.exists(path):
                print(f"⚠️  Warning: File not found: {path}")
                continue
            try:
                file_obj = self.client.files.upload(file=path)
                uploaded_files.append(file_obj)
                print(f"✅ Uploaded: {os.path.basename(path)}")
            except Exception as e:
                print(f"❌ Failed to upload {path}: {e}")
        return uploaded_files
    
    def _get_example_files(self):
        """Get or upload example files, using a persistent disk cache."""
        # 1. Try to load from in-memory cache first
        if self._pos_files and self._neg_files:
            return self._pos_files, self._neg_files

        # 2. Try to load from disk cache
        if self.cache_file.exists():
            print(f"♻️ Attempting to reuse cached files from: {self.cache_file}")
            with open(self.cache_file, 'r') as f:
                try:
                    cached_data = json.load(f)
                    # Check if cache is expired (older than 48 hours)
                    cache_time = datetime.datetime.fromisoformat(cached_data['timestamp'])
                    if datetime.datetime.now() - cache_time < datetime.timedelta(hours=48):
                        # Validate required fields exist in cache entries
                        def valid(entry):
                            return all(k in entry and entry[k] for k in ("name", "uri", "mime_type"))

                        if all(valid(e) for e in cached_data.get('pos_files', [])) and all(valid(e) for e in cached_data.get('neg_files', [])):
                            # Reconstruct File objects with name, uri, and mime_type
                            self._pos_files = [types.File(name=e['name'], uri=e['uri'], mime_type=e['mime_type']) for e in cached_data['pos_files']]
                            self._neg_files = [types.File(name=e['name'], uri=e['uri'], mime_type=e['mime_type']) for e in cached_data['neg_files']]
                            print("✅ Successfully reused cached example files.")
                            return self._pos_files, self._neg_files
                        else:
                            print("⚠️ Cache missing required fields (name, uri, mime_type). Re-uploading...")
                    else:
                        print("⚠️ Cache expired. Re-uploading...")
                except (json.JSONDecodeError, KeyError):
                    print("⚠️ Invalid cache file. Re-uploading...")

        # 3. If all else fails, upload the files and create the cache
        print("📤 Uploading new example images...")
        self._pos_files = self._upload_many(self.pos_paths)
        self._neg_files = self._upload_many(self.neg_paths)
        print(f"✅ Uploaded {len(self._pos_files)} positive and {len(self._neg_files)} negative examples.")

        # Save the new file handles (name and uri) to the cache
        def guess_mime_from_name(name: str) -> str:
            # Try to guess mime type from the filename extension; default to image/jpeg
            mt, _ = mimetypes.guess_type(name)
            return mt or 'image/jpeg'

        cache_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'pos_files': [{'name': f.name, 'uri': f.uri, 'mime_type': getattr(f, 'mime_type', None) or guess_mime_from_name(f.name)} for f in self._pos_files],
            'neg_files': [{'name': f.name, 'uri': f.uri, 'mime_type': getattr(f, 'mime_type', None) or guess_mime_from_name(f.name)} for f in self._neg_files],
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"💾 Saved new file handles to cache: {self.cache_file}")

        return self._pos_files, self._neg_files
    
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((exceptions.ResourceExhausted, exceptions.ServiceUnavailable)),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        before_sleep=tenacity.before_sleep_log(logging.getLogger(__name__), logging.INFO),
    )
    def _generate_content_with_retry(self, **kwargs):
        """Wraps the generate_content call with tenacity's retry logic."""
        return self.client.models.generate_content(**kwargs)

    def classify_image(self, target_image_path):
        """
        Classify a heat protection sleeve image using few-shot learning
        
        Args:
            target_image_path (str): Path to the image to classify
            
        Returns:
            str: JSON response from Gemini API
        """
        # Validate target image
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"Target image not found: {target_image_path}")
        
        print(f"🔍 Classifying: {os.path.basename(target_image_path)}")
        
        # Get example files
        pos_files, neg_files = self._get_example_files()
        
        # Upload target image
        try:
            target_file = self.client.files.upload(file=target_image_path)
            print(f"✅ Uploaded target image: {os.path.basename(target_image_path)}")
        except Exception as e:
            raise Exception(f"Failed to upload target image: {e}")
        
        # Build examples with labels
        examples_parts = []
        
        # Add positive examples
        for f in pos_files:
            examples_parts += [f, "Label: POSITIVE (OK example)"]
        
        # Add negative examples
        for f in neg_files:
            examples_parts += [f, "Label: NEGATIVE (Broken example)"]
        
        # Generate classification with retry logic
        try:
            print("🤖 Generating classification...")
            response = self._generate_content_with_retry(
                model="gemini-2.5-pro",
                contents=[
                    *examples_parts,             # Example images with labels
                    "TARGET image follows:",     # Separator
                    target_file,                 # Image to classify
                    USER_PROMPT,                 # Classification instructions
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                ),
            )
            
            print("✅ Classification completed!")
            return response.text
            
        except Exception as e:
            raise Exception(f"Classification failed after multiple retries: {e}")
    
    def classify_batch(self, image_paths):
        """
        Classify multiple images
        
        Args:
            image_paths (list): List of image paths to classify
            
        Returns:
            dict: Dictionary mapping image paths to classification results
        """
        results = {}
        
        print(f"🔄 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n--- Processing {i}/{len(image_paths)} ---")
            try:
                result = self.classify_image(image_path)
                results[image_path] = result
            except Exception as e:
                print(f"❌ Failed to process {image_path}: {e}")
                results[image_path] = f"ERROR: {e}"
        
        return results

# Main function for command-line usage
def main():
    """Main function for command-line usage"""
    import argparse

    print("🔧 Heat Protection Sleeve Classifier")
    print("=" * 50)

    parser = argparse.ArgumentParser(description="Classify HPS images (single image or grouped engines).")
    parser.add_argument("--target", "-t", help="Path to a single image (legacy single-image mode).")
    parser.add_argument("--directories", "-d", nargs="+",
                        default=["Archiv/correct", "Archiv/notcorrect"],
                        help="Test directories with engine images (multi-view engines).")
    parser.add_argument("--train-correct", default="Archiv/correct_training",
                        help="Directory with OK training engines (multi-view).")
    parser.add_argument("--train-notcorrect", default="Archiv/notcorrect_training",
                        help="Directory with Broken training engines (multi-view).")
    parser.add_argument("--batch-mode", action="store_true",
                        help="If set, run engine-level batch classification using directories.")
    args = parser.parse_args()

    # Initialize classifier
    try:
        classifier = HeatProtectionClassifier()
        print("✅ Classifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return 1

    if args.batch_mode:
        # Engine-level classification using dirs
        results = classifier.classify_grouped_engines(
            test_dirs=args.directories,
            train_correct_dir=args.train_correct,
            train_notcorrect_dir=args.train_notcorrect
        )
        print("\n" + "=" * 50)
        print("🎯 ENGINE-LEVEL RESULTS (JSON lines):")
        print("=" * 50)
        for r in results:
            print(json.dumps(r, ensure_ascii=False))
        print("\n✨ Engine-batch classification completed!")
        return 0

    # Legacy single-image mode
    target_image = args.target or "Archiv/notcorrect/145700277_image_2.jpeg"
    if not args.target:
        print(f"ℹ️  No target image specified, using default: {target_image}")

    try:
        result = classifier.classify_image(target_image)
        print("\n" + "=" * 50)
        print("🎯 CLASSIFICATION RESULT (single image):")
        print("=" * 50)
        print(result)
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return 1

    print("\n✨ Classification completed successfully!")
    return 0

