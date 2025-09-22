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

# System Prompt (can be customized here)
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
