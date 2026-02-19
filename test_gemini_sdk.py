#!/usr/bin/env python3
"""
Quick test script to validate google-genai SDK integration
Tests both text and image generation configurations
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def test_sdk_initialization():
    """Test that the SDK can be initialized"""
    print("🧪 Testing SDK Initialization...")

    if not GOOGLE_API_KEY:
        print("  ⚠️  GOOGLE_API_KEY not found in .env")
        return False

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        print("  ✓ Client initialized successfully")
        return True
    except Exception as e:
        print(f"  ❌ Client initialization failed: {e}")
        return False


def test_text_config():
    """Test text generation config structure"""
    print("\n🧪 Testing Text Generation Config...")

    try:
        config = types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=2048
        )
        print("  ✓ Text config created successfully")
        print(f"    Temperature: {config.temperature}")
        print(f"    Max tokens: {config.max_output_tokens}")
        return True
    except Exception as e:
        print(f"  ❌ Text config failed: {e}")
        return False


def test_image_config():
    """Test image generation config structure"""
    print("\n🧪 Testing Image Generation Config...")

    try:
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
                image_size="4K"
            )
        )
        print("  ✓ Image config created successfully")
        print(f"    Response modalities: {config.response_modalities}")
        print(f"    Thinking enabled: {config.thinking_config.include_thoughts}")
        print(f"    Aspect ratio: {config.image_config.aspect_ratio}")
        print(f"    Image size: {config.image_config.image_size}")
        return True
    except Exception as e:
        print(f"  ❌ Image config failed: {e}")
        return False


def test_simple_text_generation():
    """Test a simple text generation call"""
    print("\n🧪 Testing Simple Text Generation (actual API call)...")

    if not GOOGLE_API_KEY:
        print("  ⚠️  Skipped - No API key")
        return False

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=50
        )

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Say 'Hello from Gemini SDK!' in a creative way (max 10 words)",
            config=config
        )

        # Debug: print response structure
        print(f"  📋 Response type: {type(response)}")
        print(f"  📋 Has candidates: {hasattr(response, 'candidates')}")

        if hasattr(response, 'candidates') and response.candidates:
            print(f"  📋 Number of candidates: {len(response.candidates)}")
            candidate = response.candidates[0]
            print(f"  📋 Candidate type: {type(candidate)}")

            if hasattr(candidate, 'content') and candidate.content:
                print(f"  📋 Has content: True")
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    print(f"  📋 Number of parts: {len(candidate.content.parts)}")
                    part = candidate.content.parts[0]
                    if hasattr(part, 'text'):
                        text = part.text.strip()
                        print(f"  ✓ API call successful")
                        print(f"    Response: {text}")
                        return True

        print("  ⚠️  Response structure unexpected - but API is reachable")
        print(f"    This may indicate the model name needs adjustment")
        return False

    except Exception as e:
        print(f"  ❌ API call failed: {e}")
        import traceback
        print(f"    Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Google GenAI SDK Integration Test")
    print("=" * 70)

    results = []

    results.append(("SDK Initialization", test_sdk_initialization()))
    results.append(("Text Config", test_text_config()))
    results.append(("Image Config", test_image_config()))
    results.append(("Simple Text Generation", test_simple_text_generation()))

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! SDK integration is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check configuration.")
