#!/usr/bin/env python3
"""
Heat Protection Sleeve Segmentation
===================================

This script uses Google Gemini Vision API to segment heat protection sleeves
in automotive engine images, providing detailed masks and overlays.

Usage:
    python heat_protection_segmentation.py [image_path] [--output-dir OUTPUT_DIR]
"""

import os
import json
import base64
import io
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple

from google import genai
from google.genai import types

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# The API key is now loaded from your .env file
# It will look for a variable named API_KEY_2
API_KEY = os.getenv("API_KEY_2")

class HeatProtectionSegmenter:
    def __init__(self, api_key: str = None):
        """Initialize the heat protection sleeve segmenter"""
        # Use the provided api_key or fall back to the one from the .env file
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("API_KEY_2 not found in .env file or not provided.")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Segmentation prompt specifically for heat protection sleeves
        self.segmentation_prompt = """
        Analyze this automotive engine image and provide segmentation masks for heat protection components.
        Focus specifically on:
        
        1. Heat Protection Sleeves (HPS) - protective sheaths around hoses/wires near hot components
        2. Hot Zone Components - exhaust manifolds, turbo housings, EGR pipes that generate heat
        3. Protected Lines - hoses, wires, or conduits that have or need heat protection
        4. Fasteners - clamps, zip ties, or other retention mechanisms for sleeves
        
        For each identified component, determine:
        - Type: "heat_protection_sleeve", "hot_zone_component", "protected_line", "fastener"
        - Condition: "good", "damaged", "missing", "displaced"
        - Material: "reflective_foil", "braided_fiberglass", "textile", "black_sleeve", "metal", "rubber", "plastic"
        - Heat_risk: "high", "medium", "low" (proximity to heat sources)
        
        Output a JSON list of segmentation masks where each entry contains:
        - "box_2d": 2D bounding box coordinates [y0, x0, y1, x1] normalized to 1000
        - "mask": segmentation mask as base64 PNG data
        - "label": descriptive label (e.g., "damaged_heat_sleeve", "exhaust_manifold", "unprotected_hose")
        - "type": component type from the list above
        - "condition": condition assessment
        - "material": material type
        - "heat_risk": heat risk level
        - "confidence": confidence score 0.0-1.0
        
        Be thorough in identifying all heat protection elements and potential risk areas.
        """
    
    def parse_json_response(self, json_output: str) -> List[Dict[str, Any]]:
        """Parse JSON response from Gemini, handling markdown formatting"""
        try:
            # Remove markdown fencing if present
            lines = json_output.splitlines()
            json_text = json_output
            
            for i, line in enumerate(lines):
                if line.strip() == "```json":
                    json_text = "\n".join(lines[i+1:])
                    json_text = json_text.split("```")[0]
                    break
            
            # Parse JSON
            items = json.loads(json_text)
            
            # Ensure it's a list
            if isinstance(items, dict):
                items = [items]
            
            return items
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Failed to parse JSON response: {e}")
            print(f"Raw response: {json_output[:500]}...")
            return []
    
    def validate_segmentation_item(self, item: Dict[str, Any]) -> bool:
        """Validate that a segmentation item has required fields"""
        required_fields = ["box_2d", "mask", "label"]
        
        for field in required_fields:
            if field not in item:
                print(f"⚠️  Missing required field '{field}' in segmentation item")
                return False
        
        # Validate bounding box
        box = item["box_2d"]
        if not isinstance(box, list) or len(box) != 4:
            print(f"⚠️  Invalid bounding box format: {box}")
            return False
        
        # Validate mask format
        mask_data = item["mask"]
        if not isinstance(mask_data, str):
            print(f"⚠️  Mask data is not a string")
            return False
        
        # A valid mask can be the old base64 PNG or the new, shorter RLE string.
        # We will perform a basic check here. A more robust check happens in process_mask.
        if not mask_data.startswith("data:image/png;base64,") and len(mask_data) > 1000:
             print(f"⚠️  Mask format is not a recognized base64 PNG or a short RLE string.")
             return False
        
        return True
    
    def decode_rle_mask(self, rle_string: str, shape: Tuple[int, int]) -> Image.Image:
        """Decodes a Google AI RLE mask string into a PIL Image."""
        binary_string = base64.b64decode(rle_string)
        
        # Unpack the binary string
        counts = np.frombuffer(binary_string, dtype=np.uint32)
        
        # Create the mask array
        mask_array = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        
        # Fill the mask array using RLE counts
        current_pos = 0
        current_val = 0
        for count in counts:
            mask_array[current_pos:current_pos + count] = current_val
            current_pos += count
            current_val = 1 - current_val
        
        # Reshape and create image
        mask_array = mask_array.reshape(shape)
        return Image.fromarray(mask_array * 255, mode='L')

    def process_mask(self, mask_data: str, box: List[int], image_size: Tuple[int, int]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Process mask data (RLE or base64) and return PIL Image with coordinates"""
        
        try:
            # Convert normalized coordinates to pixel coordinates
            y0 = int(box[0] / 1000 * image_size[1])
            x0 = int(box[1] / 1000 * image_size[0])
            y1 = int(box[2] / 1000 * image_size[1])
            x1 = int(box[3] / 1000 * image_size[0])

            # Validate coordinates
            if y0 >= y1 or x0 >= x1 or x0 < 0 or y0 < 0 or x1 > image_size[0] or y1 > image_size[1]:
                print(f"⚠️  Invalid coordinates: ({x0}, {y0}) to ({x1}, {y1}) for image size {image_size}")
                return None, None

            mask_width = x1 - x0
            mask_height = y1 - y0

            # Check if the mask is RLE or base64
            if mask_data.startswith("data:image/png;base64,"):
                # Handle legacy base64 PNG format
                png_str = mask_data.removeprefix("data:image/png;base64,")
                mask_bytes = base64.b64decode(png_str)
                mask = Image.open(io.BytesIO(mask_bytes))
                mask = mask.resize((mask_width, mask_height), Image.Resampling.BILINEAR)
            else:
                # Handle new RLE format
                mask = self.decode_rle_mask(mask_data, (mask_height, mask_width))

            return mask, (x0, y0, x1, y1)

        except Exception as e:
            print(f"❌ Error processing mask: {e}")
            return None, None
    
    def create_colored_overlay(self, image: Image.Image, mask: Image.Image, 
                             coords: Tuple[int, int, int, int], 
                             item_info: Dict[str, Any]) -> Image.Image:
        """Create colored overlay based on component type and condition"""
        
        # Color mapping for different component types and conditions
        color_map = {
            "heat_protection_sleeve": {
                "good": (0, 255, 0, 120),      # Green - good sleeve
                "damaged": (255, 165, 0, 120), # Orange - damaged sleeve
                "missing": (255, 0, 0, 120),   # Red - missing sleeve
                "displaced": (255, 255, 0, 120) # Yellow - displaced sleeve
            },
            "hot_zone_component": (255, 0, 0, 80),     # Red - heat source
            "protected_line": (0, 0, 255, 80),         # Blue - protected line
            "fastener": (128, 0, 128, 80),             # Purple - fastener
            "default": (255, 255, 255, 100)            # White - unknown
        }
        
        # Get color based on type and condition
        component_type = item_info.get("type", "default")
        condition = item_info.get("condition", "good")
        
        if component_type in color_map and isinstance(color_map[component_type], dict):
            color = color_map[component_type].get(condition, color_map["default"])
        else:
            color = color_map.get(component_type, color_map["default"])
        
        # Create overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        x0, y0, x1, y1 = coords
        mask_array = np.array(mask.convert('L'))
        
        # Apply mask with color
        for y in range(y0, min(y1, image.size[1])):
            for x in range(x0, min(x1, image.size[0])):
                mask_y = y - y0
                mask_x = x - x0
                
                if (mask_y < mask_array.shape[0] and mask_x < mask_array.shape[1] and 
                    mask_array[mask_y, mask_x] > 128):
                    overlay_draw.point((x, y), fill=color)
        
        return overlay
    
    def add_annotations(self, image: Image.Image, items: List[Dict[str, Any]]) -> Image.Image:
        """Add text annotations to the image"""
        
        # Create a copy for annotation
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        for i, item in enumerate(items):
            if not self.validate_segmentation_item(item):
                continue
            
            # Get bounding box
            box = item["box_2d"]
            y0 = int(box[0] / 1000 * image.size[1])
            x0 = int(box[1] / 1000 * image.size[0])
            y1 = int(box[2] / 1000 * image.size[1])
            x1 = int(box[3] / 1000 * image.size[0])
            
            # Create label text
            label = item.get("label", f"Item_{i}")
            condition = item.get("condition", "")
            confidence = item.get("confidence", 0.0)
            
            text = f"{label}"
            if condition:
                text += f" ({condition})"
            if confidence > 0:
                text += f" {confidence:.2f}"
            
            # Draw bounding box
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            
            # Draw label background
            text_bbox = draw.textbbox((0, 0), text, font=small_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            label_y = max(0, y0 - text_height - 5)
            draw.rectangle([x0, label_y, x0 + text_width + 4, label_y + text_height + 4], 
                         fill="red", outline="red")
            
            # Draw text
            draw.text((x0 + 2, label_y + 2), text, fill="white", font=small_font)
        
        return annotated
    
    def segment_image(self, image_path: str, output_dir: str = "segmentation_outputs") -> Dict[str, Any]:
        """
        Segment heat protection sleeves in an image
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save segmentation results
            
        Returns:
            Dictionary containing segmentation results and metadata
        """
        
        print(f"🔍 Segmenting heat protection sleeves in: {Path(image_path).name}")
        
        # Load and prepare image
        try:
            image = Image.open(image_path)
            original_size = image.size
            
            # Resize for processing (Gemini works better with smaller images)
            image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            print(f"📐 Image resized from {original_size} to {image.size}")
            
        except Exception as e:
            raise Exception(f"Failed to load image: {e}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate segmentation
        try:
            print("🤖 Generating segmentation masks...")
            
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # Better for object detection
            )
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[self.segmentation_prompt, image],
                config=config
            )
            
            print("✅ Segmentation completed!")
            
        except Exception as e:
            raise Exception(f"Segmentation failed: {e}")
        
        # Parse response
        items = self.parse_json_response(response.text)
        
        if not items:
            print("⚠️  No segmentation items found")
            return {"items": [], "output_files": []}
        
        print(f"🎯 Found {len(items)} segmented components")
        
        # Process each segmentation item
        output_files = []
        valid_items = []
        overlays = []
        
        for i, item in enumerate(items):
            if not self.validate_segmentation_item(item):
                continue
            
            # Process mask
            mask, coords = self.process_mask(item["mask"], item["box_2d"], image.size)
            
            if mask is None or coords is None:
                continue
            
            valid_items.append(item)
            
            # Create colored overlay
            overlay = self.create_colored_overlay(image, mask, coords, item)
            overlays.append(overlay)
            
            # Save individual mask
            mask_filename = f"{item['label']}_{i}_mask.png"
            mask_path = output_path / mask_filename
            mask.save(mask_path)
            output_files.append(str(mask_path))
            
            # Save individual overlay
            overlay_filename = f"{item['label']}_{i}_overlay.png"
            overlay_path = output_path / overlay_filename
            composite = Image.alpha_composite(image.convert('RGBA'), overlay)
            composite.save(overlay_path)
            output_files.append(str(overlay_path))
            
            print(f"💾 Saved {item['label']} mask and overlay")
        
        # Create combined overlay
        if overlays:
            combined_overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            
            for overlay in overlays:
                combined_overlay = Image.alpha_composite(combined_overlay, overlay)
            
            # Save combined overlay
            combined_path = output_path / f"{Path(image_path).stem}_combined_overlay.png"
            final_composite = Image.alpha_composite(image.convert('RGBA'), combined_overlay)
            final_composite.save(combined_path)
            output_files.append(str(combined_path))
            
            # Create annotated version
            annotated = self.add_annotations(final_composite.convert('RGB'), valid_items)
            annotated_path = output_path / f"{Path(image_path).stem}_annotated.png"
            annotated.save(annotated_path)
            output_files.append(str(annotated_path))
            
            print(f"💾 Saved combined overlay and annotated image")
        
        # Save segmentation data
        results = {
            "image_path": str(image_path),
            "original_size": original_size,
            "processed_size": image.size,
            "total_items": len(valid_items),
            "items": valid_items,
            "raw_response": response.text
        }
        
        results_path = output_path / f"{Path(image_path).stem}_segmentation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        output_files.append(str(results_path))
        
        print(f"💾 Saved segmentation results to: {results_path}")
        
        # Print summary
        self.print_segmentation_summary(valid_items)
        
        return {
            "items": valid_items,
            "output_files": output_files,
            "results_file": str(results_path)
        }
    
    def print_segmentation_summary(self, items: List[Dict[str, Any]]):
        """Print summary of segmentation results"""
        
        print("\n" + "=" * 60)
        print("🎯 SEGMENTATION SUMMARY")
        print("=" * 60)
        
        if not items:
            print("❌ No valid segmentation items found")
            return
        
        # Count by type
        type_counts = {}
        condition_counts = {}
        risk_counts = {}
        
        for item in items:
            item_type = item.get("type", "unknown")
            condition = item.get("condition", "unknown")
            risk = item.get("heat_risk", "unknown")
            
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print(f"📊 Total Components: {len(items)}")
        
        print(f"\n🔧 BY TYPE:")
        for comp_type, count in sorted(type_counts.items()):
            print(f"   {comp_type}: {count}")
        
        print(f"\n🏥 BY CONDITION:")
        for condition, count in sorted(condition_counts.items()):
            print(f"   {condition}: {count}")
        
        print(f"\n🔥 BY HEAT RISK:")
        for risk, count in sorted(risk_counts.items()):
            print(f"   {risk}: {count}")
        
        # Highlight issues
        issues = [item for item in items if item.get("condition") in ["damaged", "missing", "displaced"]]
        if issues:
            print(f"\n⚠️  ISSUES DETECTED ({len(issues)}):")
            for issue in issues:
                print(f"   - {issue['label']}: {issue.get('condition', 'unknown')}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Segment heat protection sleeves in engine images")
    parser.add_argument("image_path", nargs='?', 
                       default="Archiv/notcorrect/145700277_image_2.jpeg",
                       help="Path to the image to segment")
    parser.add_argument("--output-dir", "-o",
                       default="segmentation_outputs",
                       help="Output directory for segmentation results")
    
    args = parser.parse_args()
    
    print("🔧 Heat Protection Sleeve Segmentation")
    print("=" * 50)
    
    # Initialize segmenter
    try:
        segmenter = HeatProtectionSegmenter()
        print("✅ Segmenter initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize segmenter: {e}")
        return 1
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Image not found: {args.image_path}")
        return 1
    
    # Run segmentation
    try:
        results = segmenter.segment_image(args.image_path, args.output_dir)
        
        print(f"\n✨ Segmentation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📄 {len(results['output_files'])} files generated")
        
        return 0
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
