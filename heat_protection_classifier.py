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

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = "AIzaSyC-oEigYfUtjW0C_qNYaM2KQuI1PYqUC8Q"

# Configure logging to show tenacity's retry attempts
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# System prompt for heat protection sleeve classification
SYSTEM_PROMPT = """
Role & Objective
You are a senior automotive technician specialized in visual inspection of heat protection sleeves (HPS) in engine compartments. Your task is to classify each input image as OK, Broken, or Uncertain with a clear, evidence‑based rationale, focusing only on the heat protection sleeve and its immediate context.

1) Scope & Definitions

Target component: Heat Protection Sleeve (HPS) — a protective sheath (often reflective foil, braided fiberglass, textile, or black sleeve) that shields hoses/wires from thermal sources (e.g., exhaust manifold, turbo, EGR pipes).
Hot zone proximity: Any area close to metallic parts that typically run hot (exhaust/turbo housings, EGR piping, DPF lines). Sleeves are expected primarily in these zones.
Underlying line: The hose/wire/conduit that the sleeve protects (often rubber or polymer; may show printed part numbers).

2) Decision Classes & Core Criteria
A. OK (Healthy) — All must be true, unless otherwise noted:

Presence & Coverage: A sleeve is present where a sleeve is expected (i.e., along segments near heat sources). Coverage is continuous across the hot zone, with no major gaps exposing underlying line in the heat-adjacent segment.
Integrity: No significant fraying, tears, punctures, cracks, melted/charred spots, or open seams that expose the underlying line along the hot zone.
Positioning: Sleeve is not obviously slipped back; end terminations look intentional (trimmed/finished). Fastening (clamps/zip ties/tape wraps) appears serviceable where visible.
Surface condition: Dust, dirt, and loss of shine are acceptable (do not classify as broken for cosmetic soiling).
Accepted exceptions: Brief, intentional exposure near connectors, bends, or branching points outside the heat-critical segment is acceptable.

B. Broken (Faulty)

Missing/Displaced Sleeve: No sleeve where one is expected near a heat source or the sleeve has slid away, leaving the hot-zone segment bare.
Structural Damage: Significant fraying, tearing, holes, deep abrasions, burn/char marks, melted areas, split seams causing underlying line exposure in the hot zone.
Inadequate Coverage: Large gaps in coverage within the heat-critical segment (e.g., sleeve doesn't reach the hot metal area; gap > ~2–3 cm in hot proximity).
Fastener Failure: Missing/failed retainers/ties causing sleeve to hang loose with exposure in the heat zone.

C. Uncertain (Needs Review)

Occlusion: The relevant segment is blocked, cropped, or out of frame.
Ambiguous Coverage: Can't confirm presence/absence or condition of the sleeve in the heat-critical segment due to poor lighting, motion blur, or angle.
Context needed: It's unclear whether the segment is near a heat source (no reliable spatial cues).

Confidence policy: Output Uncertain if confidence < 0.6.

7) Output Format (Strict JSON)
Return only the following JSON (no extra commentary):

{
  "classification": "OK | Broken | Uncertain",
  "confidence": 0.0,
  "evidence_summary": "Succinct visual rationale tied to the heat zone and sleeve condition.",
  "observations": {
    "sleeve_presence": "present | absent | occluded",
    "coverage_in_heat_zone": "continuous | partial_gap | absent | uncertain",
    "integrity": ["no_damage", "fray", "tear", "hole", "burn_char", "melted", "split_seam", "unknown"],
    "positioning": ["well_positioned", "slipped_back", "loose_end", "missing_fastener", "unknown"],
    "hot_zone_cues": ["exhaust_metal_nearby", "turbo_housing", "egr_pipe", "none_visible", "occluded"],
    "cosmetics": ["dusty", "clean", "oily", "glare", "shadowed"]
  },
  "roi_notes": "Describe where on the image the heat zone and sleeve were inspected (landmarks, relative positions).",
  "pitfall_checks": ["distinguish_cosmetic_vs_structural", "text_on_hose_not_conclusive", "angle_occlusion_checked", "component_mis-ID_checked"],
  "needs_followup": false,
  "followup_recommendations": "If Uncertain or low confidence, specify desired angle/zoom/lighting."
}
"""

USER_PROMPT = "Classify the heat protection sleeve in this engine photo. Focus on the hot-zone segment. Return only the JSON as specified. If Uncertain, tell me exactly which angle/area to re-capture."

class HeatProtectionClassifier:
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

def main():
    """Main function for command-line usage"""
    print("🔧 Heat Protection Sleeve Classifier")
    print("=" * 50)
    
    # Initialize classifier
    try:
        classifier = HeatProtectionClassifier()
        print("✅ Classifier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return 1
    
    # Determine target image
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        # Use default test image
        target_image = "Archiv/notcorrect/145700277_image_2.jpeg"
        print(f"ℹ️  No target image specified, using default: {target_image}")
    
    # Classify image
    try:
        result = classifier.classify_image(target_image)
        print("\n" + "=" * 50)
        print("🎯 CLASSIFICATION RESULT:")
        print("=" * 50)
        print(result)
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return 1
    
    print("\n✨ Classification completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
