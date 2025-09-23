#!/usr/bin/env python3
"""
Configuration file for Heat Protection Sleeve Classification Project
==================================================================
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Configuration ---
# The API key is now loaded from your .env file
# It will look for a variable named API_KEY_2
API_KEY = os.getenv("API_KEY_2")
MODEL_NAME = "gemini-2.5-pro"  # or "gemini-2.5-flash" for faster processing

# Directory Structure
PROJECT_ROOT = Path(__file__).parent
ARCHIV_DIR = PROJECT_ROOT / "Archiv"
CORRECT_DIR = ARCHIV_DIR / "correct"
NOTCORRECT_DIR = ARCHIV_DIR / "notcorrect"
RESULTS_DIR = PROJECT_ROOT / "results"

# Example Images for Few-Shot Learning
POSITIVE_EXAMPLES = [
    "Archiv/correct/147368364_image_2.jpeg",
    "Archiv/correct/147457017_image_2.jpeg", 
    "Archiv/correct/147533541_image_1.jpeg",
]

NEGATIVE_EXAMPLES = [
    "Archiv/notcorrect/145010971_image_1.jpeg",
    "Archiv/notcorrect/145322610_image_2.jpeg",
    "Archiv/notcorrect/145623226_image_2.jpeg",
]

# Classification Settings
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for certain classification
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Analysis Settings
VISUALIZATION_DPI = 300
FIGURE_SIZE = (12, 8)

# Ground Truth Mapping
GROUND_TRUTH_MAPPING = {
    "correct": "OK",
    "notcorrect": "Broken"
}

# Set threshold for uncertain classification
UNCERTAIN_THRESHOLD = 0.6

# System Prompt (can be customized here)
SYSTEM_PROMPT = f"""
Role & Objective
You are a senior automotive technician specialized in visual inspection of heat protection sleeves (HPS) in engine compartments. Your task is to classify each input image as OK, Broken, or Uncertain with a clear, evidence‑based rationale, focusing only on the heat protection sleeve and its immediate context.

1) Scope & Definitions

Target component: Heat Protection Sleeve (HPS) — a protective sheath (often reflective foil, braided fiberglass, textile, or black sleeve) that shields hoses/wires from thermal sources (e.g., exhaust manifold, turbo, EGR pipes).
Hot zone proximity: Any area close to metallic parts that typically run hot (exhaust/turbo housings, EGR piping, DPF lines). Sleeves are expected primarily in these zones.
Underlying line: The hose/wire/conduit that the sleeve protects (often rubber or polymer; may show printed part numbers).

2) Decision Classes & Core Criteria
A. OK (Healthy) — All must be true, unless otherwise noted:
- Presence & Coverage: A sleeve is present where expected and coverage is continuous across the hot zone (no major gaps).
- Integrity: No significant fraying/tears/holes/melted/charred areas or split seams exposing the line in the hot zone.
- Positioning: Not obviously slipped back; terminations intentional; fasteners serviceable.
- Surface condition: Cosmetic soiling is acceptable.
- Accepted exceptions: Brief intentional exposure near connectors/bends/branches outside heat-critical segment.

B. Broken (Faulty)
- Missing/Displaced Sleeve in hot zone.
- Structural Damage: fraying/tears/holes/burn/melted/split seams exposing the line in the hot zone.
- Inadequate Coverage: large gaps in heat-critical segment (e.g., gap > ~2–3 cm).
- Fastener Failure causing exposure in the heat zone.

C. Uncertain (Needs Review)
- Occlusion/cropping/out-of-frame of the relevant segment.
- Ambiguous coverage due to poor lighting/blur/angle.
- Insufficient context to confirm hot zone proximity.

Confidence policy: If your classification confidence is below {UNCERTAIN_THRESHOLD:.2f}, return "Uncertain".

7) Output Format (Strict JSON)
Return only the following JSON (no extra commentary):

{{
  "classification": "OK" | "Broken" | "Uncertain",
  "confidence": 0.0,
  "evidence_summary": "Succinct visual rationale tied to the heat zone and sleeve condition.",
  "observations": {{
    "sleeve_presence": "present" | "absent" | "occluded",
    "coverage_in_heat_zone": "continuous" | "partial_gap" | "absent" | "uncertain",
    "integrity": ["no_damage", "fray", "tear", "hole", "burn_char", "melted", "split_seam", "unknown"],
    "positioning": ["well_positioned", "slipped_back", "loose_end", "missing_fastener", "unknown"],
    "hot_zone_cues": ["exhaust_metal_nearby", "turbo_housing", "egr_pipe", "none_visible", "occluded"],
    "cosmetics": ["dusty", "clean", "oily", "glare", "shadowed"]
  }},
  "roi_notes": "Describe where on the image the heat zone and sleeve were inspected (landmarks, relative positions).",
  "pitfall_checks": ["distinguish_cosmetic_vs_structural", "text_on_hose_not_conclusive", "angle_occlusion_checked", "component_mis-ID_checked"],

  // NEW: Uncertainty section (REQUIRED if classification == "Uncertain" OR confidence < threshold)
  "why_uncertain": {{
    "reasons": ["occlusion" | "poor_lighting" | "motion_blur" | "insufficient_context" | "ambiguous_coverage" | "low_resolution" | "other"],
    "narrative": "Brief description explaining why a confident decision was not possible."
  }},

  // NEW: Best guess even when uncertain
  "best_guess": "OK" | "Broken",
  "best_guess_confidence": 0.0,

  // NEW: Ask instead of assuming
  "missing_information_questions": [
    "Concrete, answerable questions (e.g., 'Is the metallic pipe in the upper-right an exhaust component?')"
  ],

  // Follow-up flag and recommendations
  "needs_followup": false,
  "followup_recommendations": "If Uncertain or low confidence, specify desired angle/zoom/lighting."
}}

Validation rules:
- If classification == "Uncertain" OR confidence < {UNCERTAIN_THRESHOLD:.2f}:
  - why_uncertain.reasons MUST be non-empty
  - best_guess and best_guess_confidence MUST be provided
  - needs_followup MUST be true
  - missing_information_questions MUST list concrete questions (do not assume answers)
- best_guess_confidence MUST be <= confidence if classification is not "Uncertain" (i.e., best_guess is only relevant when uncertain).

Return only the JSON object described above.
"""


USER_PROMPT = "Classify the heat protection sleeve in this engine photo. Focus on the hot-zone segment. Return only the JSON as specified. If Uncertain, tell me exactly which angle/area to re-capture."

def ensure_directories():
    """Create necessary directories if they don't exist"""
    RESULTS_DIR.mkdir(exist_ok=True)
    
def validate_config():
    """Validate configuration settings"""
    issues = []
    
    if not API_KEY or API_KEY == "your_api_key_here":
        issues.append("API_KEY not set properly")
    
    if not ARCHIV_DIR.exists():
        issues.append(f"Archive directory not found: {ARCHIV_DIR}")
    
    # Check example files
    for example_file in POSITIVE_EXAMPLES + NEGATIVE_EXAMPLES:
        if not (PROJECT_ROOT / example_file).exists():
            issues.append(f"Example file not found: {example_file}")
    
    return issues

if __name__ == "__main__":
    print("🔧 Configuration Validation")
    print("=" * 30)
    
    ensure_directories()
    issues = validate_config()
    
    if issues:
        print("❌ Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Configuration is valid!")
        
    print(f"\n📁 Project Structure:")
    print(f"   Root: {PROJECT_ROOT}")
    print(f"   Archive: {ARCHIV_DIR}")
    print(f"   Results: {RESULTS_DIR}")
    print(f"   Positive Examples: {len(POSITIVE_EXAMPLES)}")
    print(f"   Negative Examples: {len(NEGATIVE_EXAMPLES)}")
