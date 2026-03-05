#!/usr/bin/env python3
"""
Compare Gemini image generation models:
- Nano Banana Pro (gemini-3-pro-image-preview) - Professional quality with thinking
- Nano Banana 2 (gemini-3.1-flash-image-preview) - High-efficiency, faster generation

Generates the same prompt with both models to compare quality and speed.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Models to compare
NANO_BANANA_PRO = "gemini-3-pro-image-preview"
NANO_BANANA_2 = "gemini-3.1-flash-image-preview"

# Test prompt - simple but challenging enough to see quality differences
TEST_PROMPT = """Create a vibrant podcast cover art featuring a cartoonish lightning bolt character with a friendly smile.
The character should have:
- Bright yellow coloring with electric blue highlights
- Simple geometric shapes
- A cheerful, energetic personality
- Clean, professional illustration style

Background: Deep purple gradient with subtle geometric patterns.
Text: "TEST" in bold, modern font at the top.

Style: Modern, clean, professional podcast cover art."""


def load_reference_images():
    """Load the bolt reference image"""
    references = []

    bolt_path = Path("references/bolt.jpg")
    if bolt_path.exists():
        with open(bolt_path, "rb") as f:
            references.append({
                "mime_type": "image/jpeg",
                "data": f.read()
            })
        print(f"  ✓ Loaded reference: {bolt_path}")
    else:
        print(f"  ⚠️  Reference not found: {bolt_path}")

    return references


def generate_with_model(client, model_name, prompt, references, output_path):
    """Generate image with specified model and save it"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    try:
        # Build content with references
        content_parts = []
        for ref in references:
            content_parts.append(types.Part.from_bytes(
                data=ref["data"],
                mime_type=ref["mime_type"]
            ))
        content_parts.append(prompt)

        # Configure generation
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
                image_size="4K"
            )
        )

        # Time the generation
        start_time = time.time()

        response = client.models.generate_content(
            model=model_name,
            contents=content_parts,
            config=config
        )

        elapsed_time = time.time() - start_time

        # Extract and save image
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Save image
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(part.inline_data.data)

                    print(f"  ✓ Generated in {elapsed_time:.2f}s")
                    print(f"  ✓ Saved to: {output_path}")
                    return elapsed_time
                elif hasattr(part, 'text') and part.text:
                    # Print any thinking/reasoning
                    if len(part.text) > 200:
                        print(f"  💭 Model reasoning: {part.text[:200]}...")
                    else:
                        print(f"  💭 Model reasoning: {part.text}")

        print(f"  ❌ No image generated")
        return None

    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        import traceback
        print(f"     {traceback.format_exc()}")
        return None


def main():
    print("="*70)
    print("Gemini Model Comparison Test")
    print("="*70)

    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not found in .env")
        return

    # Initialize client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Load references
    print("\n📸 Loading reference images...")
    references = load_reference_images()

    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Test both models
    results = {}

    # Test Nano Banana Pro (current model)
    pro_output = output_dir / "nano_banana_pro.jpg"
    pro_time = generate_with_model(
        client,
        NANO_BANANA_PRO,
        TEST_PROMPT,
        references,
        pro_output
    )
    if pro_time:
        results[NANO_BANANA_PRO] = pro_time

    # Test Nano Banana 2 (new faster model)
    flash_output = output_dir / "nano_banana_2.jpg"
    flash_time = generate_with_model(
        client,
        NANO_BANANA_2,
        TEST_PROMPT,
        references,
        flash_output
    )
    if flash_time:
        results[NANO_BANANA_2] = flash_time

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    if results:
        for model, elapsed in results.items():
            model_label = "Nano Banana Pro" if "pro" in model else "Nano Banana 2"
            print(f"{model_label:20} - {elapsed:6.2f}s")

        if len(results) == 2:
            pro_time = results.get(NANO_BANANA_PRO)
            flash_time = results.get(NANO_BANANA_2)
            if pro_time and flash_time:
                speedup = pro_time / flash_time
                print(f"\n⚡ Nano Banana 2 is {speedup:.2f}x faster")

        print(f"\n📁 Compare outputs in: {output_dir}/")
        print(f"   - nano_banana_pro.jpg  (current model)")
        print(f"   - nano_banana_2.jpg    (new faster model)")
    else:
        print("❌ No successful generations to compare")


if __name__ == "__main__":
    main()
