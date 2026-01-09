#!/usr/bin/env python3
"""
Podcast Cover Image Generator for The Cloud Pod
Generates creative podcast cover art using OpenAI and Google Gemini APIs
with automated text overlays and logo compositing.
"""

import asyncio
import aiohttp
import argparse
import base64
import json
import os
import requests
import sys
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_IMAGE_ENDPOINT = "https://api.openai.com/v1/images/generations"
GEMINI_TEXT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_IMAGE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"

# Image Dimensions
SQUARE_SIZE = (3000, 3000)  # Podcast cover format
SOCIAL_SIZE = (1200, 630)   # Open Graph format

# Typography Configuration - Professional grouped design
SQUARE_EPISODE_FONT_SIZE = 100  # Episode number size
SQUARE_TITLE_FONT_SIZE = 48     # Title size
SOCIAL_EPISODE_FONT_SIZE = 70   # Episode number for social
SOCIAL_TITLE_FONT_SIZE = 32     # Title for social

SQUARE_LOGO_SIZE = (300, 300)
SOCIAL_LOGO_SIZE = (160, 160)

# Protected area for title/logo (overlay bar at bottom)
SQUARE_BAR_HEIGHT = 650   # Height of overlay bar for square images
SOCIAL_BAR_HEIGHT = 160   # Height of overlay bar for social images
TITLE_BAR_ALPHA = 160     # Semi-transparent black bar (0=transparent, 255=opaque) - lighter for visibility
TITLE_BAR_PADDING = 50    # Padding inside the bar

# Text spacing
SQUARE_LINE_SPACING = 25         # Line spacing for title in square format
SQUARE_EPISODE_TITLE_GAP = 40    # Gap between episode number and title in square format
SOCIAL_LINE_SPACING = 10         # Line spacing for title in social format
SOCIAL_EPISODE_TITLE_GAP = 15    # Gap between episode number and title in social format

# File Paths
SCRIPT_DIR = Path(__file__).parent
LOGO_PATH = SCRIPT_DIR / "Logo" / "smallsquare.png"
BOLT_PATH = SCRIPT_DIR / "Hosts" / "bolt.png"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Font Paths (cross-platform support)
FONT_PATHS = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    # Windows
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]

# Timeouts
OPENAI_TIMEOUT = 120
GEMINI_TIMEOUT = 180

# Rate Limiting
OPENAI_DELAY = 13  # seconds between requests (5 per minute limit)
GEMINI_CONCURRENCY = 2  # concurrent requests allowed

# Input Validation
MIN_EPISODE = 1
MAX_EPISODE = 9999
MAX_TITLE_LENGTH = 200

# File Size Limits
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB per image file

# Text Generation Tokens
CONCEPT_GENERATION_TOKENS = 300
CONCEPT_REFINEMENT_TOKENS = 500

# Image Processing
LETTERBOX_BLUR_RADIUS = 50
LETTERBOX_BRIGHTNESS = 0.5
LETTERBOX_SATURATION = 0.7

# Retry Configuration
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds


# ============================================================================
# LOGGING SETUP
# ============================================================================

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / 'cover-generator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def retry_with_backoff(func, *args, **kwargs):
    """Retry an async function with exponential backoff

    Args:
        func: Async function to retry
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Result from successful function call, or None if all retries fail
    """
    for attempt in range(MAX_API_RETRIES):
        try:
            result = await func(*args, **kwargs)
            if result is not None:
                return result
            # If result is None, retry
            logger.warning(f"Attempt {attempt + 1}/{MAX_API_RETRIES} returned None, retrying...")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < MAX_API_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1}/{MAX_API_RETRIES} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {MAX_API_RETRIES} attempts failed: {e}")
                return None
    return None


def validate_episode_number(episode: int) -> bool:
    """Validate episode number is within acceptable range

    Args:
        episode: Episode number to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(episode, int):
        logger.error(f"Episode number must be an integer, got {type(episode)}")
        return False
    if episode < MIN_EPISODE or episode > MAX_EPISODE:
        logger.error(f"Episode number must be between {MIN_EPISODE} and {MAX_EPISODE}, got {episode}")
        return False
    return True


def validate_title(title: str) -> bool:
    """Validate episode title

    Args:
        title: Episode title to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(title, str):
        logger.error(f"Title must be a string, got {type(title)}")
        return False
    if not title or not title.strip():
        logger.error("Title cannot be empty")
        return False
    if len(title) > MAX_TITLE_LENGTH:
        logger.error(f"Title too long ({len(title)} chars), max is {MAX_TITLE_LENGTH}")
        return False
    return True


def slugify_title(title: str, max_words: int = 3) -> str:
    """Convert title to slug format with first few keywords

    Example: "We Were Right (Mostly), 2026: The New Prophecies" -> "we-were-right-mostly"

    Args:
        title: Episode title to convert
        max_words: Maximum number of words to include

    Returns:
        Slug string, or "untitled" if title cannot be slugified
    """
    import re

    # Handle empty or invalid input
    if not title or not isinstance(title, str):
        logger.warning(f"Invalid title for slugification: {title}")
        return "untitled"

    # Remove punctuation and convert to lowercase
    title_clean = re.sub(r'[^\w\s-]', '', title.lower())

    # Split into words, filter empty strings, and take first max_words
    words = [w for w in title_clean.split() if w][:max_words]

    # Handle case where no valid words remain after cleaning
    if not words:
        logger.warning(f"No valid words in title after cleaning: {title}")
        return "untitled"

    # Join with hyphens and ensure it's not empty
    slug = '-'.join(words)

    # Final validation
    if not slug or slug == '-':
        logger.warning(f"Empty slug produced from title: {title}")
        return "untitled"

    return slug


def load_bolt_reference() -> Optional[str]:
    """Load Bolt mascot image and convert to base64 for API reference"""
    try:
        if not BOLT_PATH.exists():
            return None

        # Validate file size before loading
        file_size = BOLT_PATH.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            logger.warning(f"Bolt image too large ({file_size / 1024 / 1024:.1f}MB), max is {MAX_IMAGE_SIZE / 1024 / 1024}MB")
            return None

        with open(BOLT_PATH, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
    except (OSError, IOError) as e:
        logger.warning(f"Could not load Bolt reference image: {e}")
        return None


def load_host_references() -> List[str]:
    """Load all host images and convert to base64 for API reference"""
    host_refs = []
    hosts_dir = SCRIPT_DIR / "Hosts"

    try:
        # Load all host images (jpg, jpeg, png) - sorted for consistency
        host_files = []
        for host_file in hosts_dir.glob("*"):
            if host_file.suffix.lower() in ['.jpg', '.jpeg', '.png'] and host_file.name != 'bolt.png':
                host_files.append(host_file)

        # Sort by name for consistent ordering
        host_files.sort(key=lambda x: x.name)

        print(f"  📋 Loading host reference files: {[f.name for f in host_files]}")

        for host_file in host_files:
            # Validate file size before loading
            file_size = host_file.stat().st_size
            if file_size > MAX_IMAGE_SIZE:
                logger.warning(f"Host image {host_file.name} too large ({file_size / 1024 / 1024:.1f}MB), skipping")
                continue

            with open(host_file, 'rb') as f:
                image_data = f.read()
                host_refs.append(base64.b64encode(image_data).decode('utf-8'))

        print(f"  ✓ Loaded {len(host_refs)} host reference images")
        return host_refs
    except (OSError, IOError) as e:
        logger.warning(f"Could not load host reference images: {e}")
        return []


# ============================================================================
# ENUMS
# ============================================================================

class Provider(Enum):
    """AI Provider"""
    OPENAI = "openai"
    GEMINI = "gemini"


class ImageVariant(Enum):
    """Image format variant"""
    SQUARE = "square"
    SOCIAL = "social"


# ============================================================================
# STYLE GUIDE AND PROMPTS
# ============================================================================

BASE_STYLE_PROMPT = """Create a playful, professional podcast cover background with these characteristics:

ILLUSTRATION STYLE - Modern flat vector aesthetic:
- Visual style reference: Think Kurzgesagt, Slack marketing illustrations, or modern tech editorial art
- Bold shapes with clean edges and smooth gradients
- Render specific objects and characters that are described in the concept
- Playful proportions and exaggerated scale for comedic effect
- Soft shadows and depth through color variation, not harsh lighting

COLOR PALETTE:
- Primary: Blues (#0066FF), cloud theme colors
- Accent: White, light grays, strategic pops of color (yellow for Bolt's lightning bolt)
- Vibrant but professional - saturated blues, clean whites, purposeful color choices
- Use color to enhance humor and guide the eye

COMPOSITION & FRAMING:
- Primary focal point in center-to-upper area with clear visual hierarchy
- Embrace negative space - don't fill every corner
- Breathing room around characters and key objects
- Characters shown as distinct stylized figures with recognizable hair/facial hair patterns
- Foreground-midground-background depth using size and color saturation

TEXT IN SCENE:
- Include story text ONLY if mentioned in the concept (readable labels, signs, dates, etc.)
- No graphic design overlays - those are added in post-production
- Keep scene text handwritten, natural, part of the world

HUMOR & STORYTELLING:
- Embrace visual puns and literal interpretations as described in the concept
- Exaggerate proportions for comedic effect when the concept calls for it
- Add playful details that reward close inspection
- Characters express personality through pose and composition
- Render ONLY what is described in the concept - do not add extra elements"""


def build_concept_prompt(episode_title: str, previous_concepts: List[str] = None, keywords: str = None) -> str:
    """Build prompt for concept generation (text-only phase)"""

    # Add previous concepts to avoid duplicates
    previous_context = ""
    if previous_concepts and len(previous_concepts) > 0:
        previous_context = "\n\nPREVIOUS CONCEPTS ALREADY GENERATED (do NOT repeat these ideas):\n"
        for i, concept in enumerate(previous_concepts, 1):
            previous_context += f"{i}. {concept}\n"
        previous_context += "\nYour concept must be COMPLETELY DIFFERENT from these.\n"

    # Add keyword guidance if provided
    keyword_guidance = ""
    if keywords and keywords.strip():
        keyword_guidance = f"\n\nKEYWORD GUIDANCE: The user wants concepts that incorporate or emphasize these themes: {keywords.strip()}\nUse these keywords to steer your creative direction while still interpreting the episode title literally.\n"

    return f"""You are the creative director for The Cloud Pod, a tech podcast famous for visual wordplay and literal humor.

Episode Title: "{episode_title}"

YOUR TASK: Create ONE completely NEW visual scene based ONLY on "{episode_title}". Take the words in this title literally and create a specific, original scene.

APPROACH - How to interpret titles literally:
• Take individual WORDS literally (if title says "wardrobe", show actual clothing; if it says "layers", show a visual stack)
• Convert abstract concepts into PHYSICAL objects (if title mentions "conversational", show someone literally talking/yelling)
• Find the VISUAL PUN in the title's wording
• Create SPECIFIC details (not vague "tech vibes" but concrete physical objects, readable text, tangible props)
• Exaggerate for HUMOR (impossible proportions, absurd scales, playful contradictions)

CONCEPT VARIETY:
• Many strong concepts focus on OBJECTS, ENVIRONMENTS, or ABSTRACT VISUALS without any characters
• Minimalist concepts (empty server rack, floating cloud, literal interpretation of title words) are often the most effective
• Abstract/environmental concepts are encouraged
• ONLY include characters if they genuinely enhance the specific visual pun or story

IMPORTANT: Do NOT reuse visual elements from other episodes. Each concept must be completely original based on THIS episode's title.{previous_context}{keyword_guidance}

OPTIONAL CHARACTERS (use sparingly, only when truly beneficial):
• Bolt - cloud robot mascot (use only if robot/mascot fits the concept)
• The Four Hosts - podcast team (use only if concept requires people/team interaction)

Most concepts should NOT include characters. Focus on clever object-based or environmental visual metaphors first.

Return ONLY one sentence describing the specific visual scene based on "{episode_title}":"""


def build_image_prompt(concept: str, variant: ImageVariant, provider: Provider = None) -> str:
    """Build detailed prompt for image generation with optional model-specific emphasis"""

    if variant == ImageVariant.SQUARE:
        dimension_guidance = """Square format (1:1 aspect ratio).

FRAMING: Keep all important visual elements (characters, objects, focal points) in the upper 75% of the frame. The bottom quarter is reserved for title treatment in post-production. Background gradients can extend to edges.

COMPOSITION: Center-weighted or slightly upper composition. Leave bottom area visually quiet."""
    else:  # SOCIAL
        dimension_guidance = """Horizontal landscape format (roughly 16:9 aspect ratio - WIDE not tall).

FRAMING: Keep all important visual elements (characters, objects, focal points) in the upper 75% of the frame. The bottom quarter is reserved for title treatment in post-production. Background gradients can extend to edges.

COMPOSITION: Center-to-upper composition with breathing room. Leave bottom area visually quiet."""

    # Check if Bolt is mentioned in the concept
    bolt_guidance = ""
    if "bolt" in concept.lower():
        bolt_guidance = """

CHARACTER REFERENCE - "Bolt" (The Cloud Pod mascot):
Reference image provided. Match this character exactly:
- Bright saturated blue (#0066FF) cloud-shaped body with yellow lightning bolt chest
- Maintain friendly expression, simple rounded shapes, flat illustration style
- Same proportions and design as reference"""

    # Check if hosts are mentioned
    hosts_guidance = ""
    if any(word in concept.lower() for word in ["host", "hosts", "podcast hosts", "people", "silhouetted"]):
        hosts_guidance = """

SCENE COMPOSITION - The Four Podcast Hosts:
The four tech professional hosts appear in this scene (along with Bolt if mentioned).

THE FOUR HOSTS - Render as distinct, recognizable team in modern flat illustration:

Jonathan: Dark hair, clean-shaven smooth face, fuller solid build. Fresh welcoming energy.

Justin: Completely bald smooth head, silver-gray goatee (chin/mouth area), robust frame. Distinguished sage.

Matthew: Horseshoe hair pattern - bald on top with brown hair around sides and back. Scruffy full brown beard (fuller than Justin/Ryan's goatees). Slim athletic build, always smiling. Energetic optimist.

Ryan: Shortish wavy golden-brown hair, darker brown goatee (chin/mouth area, similar coverage to Justin). Medium build. Thoughtful creative presence.

VISUAL KEYS:
- Hair is the primary identifier: Justin=bald, Matthew=horseshoe, Jonathan=full dark, Ryan=wavy golden-brown
- Facial hair: Matthew has full beard; Justin and Ryan have goatees (Justin=gray, Ryan=brown)
- All light-skinned, professional attire, cohesive team aesthetic
- Each must be recognizable from their hair/facial hair signature"""

    # Model-specific style emphasis
    model_emphasis = ""
    if provider == Provider.OPENAI:
        model_emphasis = "\n\nSTYLE EMPHASIS: Prioritize clean flat illustration with bold shapes and smooth color gradients. Avoid photorealistic rendering."
    elif provider == Provider.GEMINI:
        model_emphasis = "\n\nSTYLE EMPHASIS: Match the exact visual style and character designs from any reference images provided. Maintain consistency with provided examples."

    return f"""{BASE_STYLE_PROMPT}

SPECIFIC EPISODE CONCEPT:
{concept}

CRITICAL: Render ONLY the elements described in the concept above. Do not add unrelated objects, characters, or text that are not explicitly mentioned in the concept.
{bolt_guidance}
{hosts_guidance}

COMPOSITION REQUIREMENTS:
{dimension_guidance}

Generate a background image that visualizes this concept while maintaining The Cloud Pod's professional tech aesthetic.{model_emphasis}"""


# ============================================================================
# TEXT GENERATION FUNCTIONS (Concept Phase)
# ============================================================================

async def generate_concept_openai(
    session: aiohttp.ClientSession,
    api_key: str,
    episode_title: str,
    previous_concepts: List[str] = None,
    keywords: str = None
) -> Optional[str]:
    """Generate a creative concept using OpenAI GPT-4"""

    prompt = build_concept_prompt(episode_title, previous_concepts, keywords)

    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.9,  # High creativity
        "max_tokens": 300  # Increased to match concept complexity
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        async with session.post(
            OPENAI_CHAT_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                concept = data["choices"][0]["message"]["content"].strip()
                return concept
            else:
                error_text = await response.text()
                logger.error(f"OpenAI concept generation failed: {error_text}")
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"OpenAI concept generation error: {e}")
        return None


async def generate_concept_gemini(
    session: aiohttp.ClientSession,
    api_key: str,
    episode_title: str,
    previous_concepts: List[str] = None,
    keywords: str = None
) -> Optional[str]:
    """Generate a creative concept using Google Gemini"""

    prompt = build_concept_prompt(episode_title, previous_concepts, keywords)

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.9,  # High creativity
            "maxOutputTokens": 2048  # Gemini uses ~3-4 tokens per word, need higher limit
        }
    }

    url = f"{GEMINI_TEXT_ENDPOINT}?key={api_key}"

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                concept = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return concept
            else:
                error_text = await response.text()
                logger.error(f"Gemini concept generation failed: {error_text}")
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Gemini concept generation error: {e}")
        return None


async def generate_concepts(episode_title: str) -> List[Tuple[str, str]]:
    """Generate 6 creative concepts (3 from each provider, sequentially to ensure variety)
    Returns list of tuples: (concept_text, provider_name)
    """

    print(f"\n🎨 Generating creative concepts for: \"{episode_title}\"")
    print("=" * 70)

    concepts = []
    previous_concepts = []

    async with aiohttp.ClientSession() as session:
        # Generate 3 concepts from each provider SEQUENTIALLY, each seeing previous concepts
        # This ensures variety and prevents duplicates

        providers = []
        if OPENAI_API_KEY:
            providers.extend([("OpenAI", generate_concept_openai, OPENAI_API_KEY)] * 3)
        if GOOGLE_API_KEY:
            providers.extend([("Gemini", generate_concept_gemini, GOOGLE_API_KEY)] * 3)

        for provider_name, generate_func, api_key in providers:
            print(f"  Generating {provider_name} concept {len(concepts) + 1}/6...")
            result = await retry_with_backoff(generate_func, session, api_key, episode_title, previous_concepts)

            if isinstance(result, str) and result:
                concepts.append((result, provider_name))
                previous_concepts.append(result)
            else:
                print(f"  ⚠️  {provider_name} concept generation failed")

    # Require at least 6 concepts
    if len(concepts) < 6:
        print(f"\n❌ Error: Only generated {len(concepts)} concepts, need 6")
        print("Please check API keys and try again.")
        return []

    return concepts[:6]  # Return exactly 6 concepts


async def generate_more_concepts(
    episode_title: str,
    existing_concepts: List[Tuple[str, str]],
    keywords: str = None,
    count: int = 3
) -> List[Tuple[str, str]]:
    """Generate additional concepts with optional keyword steering

    Args:
        episode_title: Episode title for context
        existing_concepts: List of (concept_text, provider_name) already generated
        keywords: Optional keywords to steer concept direction
        count: Number of additional concepts to generate (default 3)

    Returns:
        List of new (concept_text, provider_name) tuples
    """

    print(f"\n🎨 Generating {count} more concepts...")
    if keywords:
        print(f"   Keywords: {keywords}")
    print("=" * 70)

    new_concepts = []
    previous_concepts = [concept for concept, _ in existing_concepts]

    async with aiohttp.ClientSession() as session:
        # Alternate between providers for variety
        providers = []
        if OPENAI_API_KEY and GOOGLE_API_KEY:
            # If both available, alternate
            for i in range(count):
                if i % 2 == 0:
                    providers.append(("OpenAI", generate_concept_openai, OPENAI_API_KEY))
                else:
                    providers.append(("Gemini", generate_concept_gemini, GOOGLE_API_KEY))
        elif OPENAI_API_KEY:
            providers = [("OpenAI", generate_concept_openai, OPENAI_API_KEY)] * count
        elif GOOGLE_API_KEY:
            providers = [("Gemini", generate_concept_gemini, GOOGLE_API_KEY)] * count

        for provider_name, generate_func, api_key in providers:
            print(f"  Generating {provider_name} concept {len(new_concepts) + 1}/{count}...")
            result = await retry_with_backoff(
                generate_func,
                session,
                api_key,
                episode_title,
                previous_concepts,
                keywords  # Pass keywords for steering
            )

            if isinstance(result, str) and result:
                new_concepts.append((result, provider_name))
                previous_concepts.append(result)  # Add so next concept sees it
            else:
                print(f"  ⚠️  {provider_name} concept generation failed")

    if len(new_concepts) == 0:
        print(f"\n⚠️  Warning: No new concepts generated")

    return new_concepts


async def refine_concept(original_concept: str, refinement: str, episode_title: str, provider: str) -> Optional[str]:
    """Refine a concept based on user feedback (async)"""
    prompt = f"""You are refining a visual concept for a podcast cover.

Episode Title: "{episode_title}"
Original Concept: "{original_concept}"
User Refinement Request: "{refinement}"

Provide an updated concept that incorporates the user's refinement while maintaining the playful, literal interpretation style of The Cloud Pod.

Return ONLY the refined concept in one sentence:"""

    try:
        async with aiohttp.ClientSession() as session:
            if provider == "OpenAI" and OPENAI_API_KEY:
                async with session.post(
                    OPENAI_CHAT_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": CONCEPT_REFINEMENT_TOKENS
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"].strip()
            else:  # Gemini
                if GOOGLE_API_KEY:
                    url = f"{GEMINI_TEXT_ENDPOINT}?key={GOOGLE_API_KEY}"
                    async with session.post(
                        url,
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.7, "maxOutputTokens": CONCEPT_REFINEMENT_TOKENS}
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        logger.error("Refinement failed")
        return None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
        logger.error(f"Refinement error: {e}")
        return None


async def present_concepts_and_choose(concepts: List[Tuple[str, str]], episode_title: str) -> Tuple[int, str, bool]:
    """Display concepts and get user selection

    Args:
        concepts: List of (concept_text, provider_name) tuples (can grow beyond 6)
        episode_title: Episode title for refinement prompts

    Returns:
        Tuple of (selected_index, concept_text, should_regenerate)
    """

    while True:  # Outer loop to handle concept additions
        print("\n📋 Creative Concepts:")
        print("=" * 70)

        for i, (concept, provider) in enumerate(concepts, 1):
            print(f"\n{i}. [{provider}] {concept}")

        print("\n" + "=" * 70)
        concept_range = f"1-{len(concepts)}"
        refine_range = f"R1-R{len(concepts)}"
        print(f"Commands: [{concept_range}] = Select concept | 0 = Generate 6 new concepts | M = Generate MORE concepts | {refine_range} = Refine concept | X = Exit")

        while True:  # Inner loop for user input
            try:
                choice = input("\nYour choice: ").strip().upper()

                # Exit
                if choice == 'X':
                    print("\n👋 Exiting...")
                    sys.exit(0)

                # Regenerate all
                if choice == '0':
                    print("\n🔄 Regenerating concepts...")
                    return 0, "", True

                # Generate MORE concepts
                if choice == 'M':
                    keywords = input("\nOptional keywords to steer concepts (or press Enter to skip): ").strip()
                    if not keywords:
                        keywords = None

                    print(f"\nHow many additional concepts? (default: 3): ", end="")
                    count_input = input().strip()
                    count = int(count_input) if count_input and count_input.isdigit() else 3

                    new_concepts = await generate_more_concepts(episode_title, concepts, keywords, count)
                    if new_concepts:
                        concepts.extend(new_concepts)
                        print(f"\n✓ Added {len(new_concepts)} new concepts (total: {len(concepts)})")
                    break  # Break inner loop to redisplay all concepts

                # Refine concept
                if choice.startswith('R') and len(choice) > 1:
                    try:
                        refine_num = int(choice[1:])
                        if 1 <= refine_num <= len(concepts):
                            selected_concept, provider = concepts[refine_num - 1]
                            print(f"\n📝 Original concept: {selected_concept}")
                            refinement = input("\nHow would you like to refine this concept? ").strip()

                            if refinement:
                                refined = await refine_concept(selected_concept, refinement, episode_title, provider)
                                if refined:
                                    print(f"\n✨ Refined concept: {refined}\n")
                                    confirm = input("Use this refined concept? (Y/n): ").strip().lower()
                                    if confirm != 'n':
                                        return refine_num, refined, False
                            continue
                        else:
                            print(f"Please enter R1-R{len(concepts)} to refine a concept")
                            continue
                    except ValueError:
                        print(f"Please enter R# where # is 1-{len(concepts)} (e.g., R3)")
                        continue

                # Select concept
                choice_num = int(choice)
                if 1 <= choice_num <= len(concepts):
                    selected_concept, provider = concepts[choice_num - 1]
                    print(f"\n✓ Selected #{choice_num} [{provider}]: {selected_concept}\n")
                    return choice_num, selected_concept, False
                else:
                    print(f"Please enter 1-{len(concepts)}, 0, M, R#, or X")

            except ValueError:
                print(f"Please enter 1-{len(concepts)}, 0, M, R# (e.g., R3), or X")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                sys.exit(0)


# ============================================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================================

async def generate_image_openai(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    variant: ImageVariant,
    concept: str = ""
) -> Optional[bytes]:
    """Generate image using OpenAI GPT Image with reference images via edit endpoint"""

    print(f"  🎨 OpenAI generating {variant.value} variant...")

    has_bolt = "bolt" in concept.lower()
    has_hosts = any(word in concept.lower() for word in ["host", "hosts", "podcast hosts", "people", "silhouetted"])

    # Use edit endpoint ONLY if we have Bolt (actual reference image)
    # Hosts are described via text, so they don't require edit endpoint
    use_edit_endpoint = has_bolt

    if use_edit_endpoint:
        # Use images.edit endpoint with reference images
        # Build multipart form data
        import aiohttp

        form = aiohttp.FormData()
        form.add_field('model', 'gpt-image-1')
        form.add_field('prompt', prompt)
        form.add_field('size', '1024x1024')
        form.add_field('quality', 'high')
        form.add_field('input_fidelity', 'high')  # Preserve details from input images
        form.add_field('output_format', 'png')
        form.add_field('n', '1')

        # Add reference images using array syntax (image[])
        # Only add Bolt reference - hosts are described via text prompts
        if has_bolt:
            bolt_ref = load_bolt_reference()
            if bolt_ref:
                bolt_bytes = base64.b64decode(bolt_ref)
                form.add_field('image[]', bolt_bytes, filename='bolt.png', content_type='image/png')
                print("  📸 Using Bolt reference image")

        if has_hosts:
            print("  📝 Using text descriptions for 4 hosts (Jonathan, Justin, Matthew, Ryan)")

        endpoint = "https://api.openai.com/v1/images/edits"
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            async with session.post(
                endpoint,
                data=form,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=OPENAI_TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and len(data["data"]) > 0:
                        image_data = data["data"][0]
                        if "b64_json" in image_data:
                            image_bytes = base64.b64decode(image_data["b64_json"])
                            print(f"  ✓ OpenAI {variant.value} generated")
                            return image_bytes
                        elif "url" in image_data:
                            async with session.get(image_data["url"]) as img_response:
                                if img_response.status == 200:
                                    image_bytes = await img_response.read()
                                    print(f"  ✓ OpenAI {variant.value} generated")
                                    return image_bytes
                    logger.error("No image data in OpenAI response")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"OpenAI edit failed: {error_text}")
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as e:
            logger.error(f"OpenAI edit error: {e}")
            return None

    else:
        # Use generate endpoint (no Bolt reference image)
        if has_hosts:
            print("  📝 Using text descriptions for 4 hosts (Jonathan, Justin, Matthew, Ryan)")

        payload = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "high",
            "output_format": "png"
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with session.post(
                OPENAI_IMAGE_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=OPENAI_TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    if "data" in data and len(data["data"]) > 0:
                        image_data = data["data"][0]
                        if "b64_json" in image_data:
                            image_bytes = base64.b64decode(image_data["b64_json"])
                            print(f"  ✓ OpenAI {variant.value} generated")
                            return image_bytes
                        elif "url" in image_data:
                            async with session.get(image_data["url"]) as img_response:
                                if img_response.status == 200:
                                    image_bytes = await img_response.read()
                                    print(f"  ✓ OpenAI {variant.value} generated")
                                    return image_bytes

                    logger.error("No image data in OpenAI response")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"OpenAI generation failed: {error_text}")
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as e:
            logger.error(f"OpenAI generation error: {e}")
            return None


async def generate_image_gemini(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    variant: ImageVariant,
    concept: str = ""
) -> Optional[bytes]:
    """Generate image using Google Gemini with optional reference images"""

    print(f"  🎨 Gemini generating {variant.value} variant...")

    # Check for character references in concept
    parts = []
    reference_text_parts = []

    has_bolt = "bolt" in concept.lower()
    has_hosts = any(word in concept.lower() for word in ["host", "hosts", "podcast hosts", "people", "silhouetted"])

    # Load Bolt reference if mentioned
    if has_bolt:
        bolt_reference_b64 = load_bolt_reference()
        if bolt_reference_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": bolt_reference_b64
                }
            })
            reference_text_parts.append("The first image shows 'Bolt' (The Cloud Pod mascot). Match this character's exact design, colors, and proportions when Bolt appears in the scene.")
            print("  📸 Using Bolt reference image")

    # Use text descriptions for hosts instead of reference images
    if has_hosts:
        print("  📝 Using text descriptions for 4 hosts (Jonathan, Justin, Matthew, Ryan)")

    # Build final prompt with references
    if reference_text_parts:
        reference_instructions = "\n".join(reference_text_parts)
        final_prompt = f"""CRITICAL: REFERENCE IMAGES PROVIDED ABOVE

{reference_instructions}

IMPORTANT INSTRUCTIONS:
- You MUST match the exact character designs from the reference images
- For Bolt: Copy the exact colors, proportions, features, and style from the reference
- For hosts: Use the reference images as the basis for silhouettes and variations
- DO NOT reinterpret or redesign these characters
- Maintain high fidelity to the provided references

GENERATION TASK:
{prompt}"""
        parts.append({"text": final_prompt})
    else:
        parts.append({"text": prompt})

    # Gemini image generation
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"]
        }
    }

    url = f"{GEMINI_IMAGE_ENDPOINT}?key={api_key}"

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=GEMINI_TIMEOUT)
        ) as response:
            if response.status == 200:
                data = await response.json()

                # Extract image from Gemini response (match emoji system logic)
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            # Try camelCase
                            if "inlineData" in part:
                                image_data = part["inlineData"].get("data")
                                if image_data:
                                    image_bytes = base64.b64decode(image_data)
                                    print(f"  ✓ Gemini {variant.value} generated")
                                    return image_bytes
                            # Try snake_case
                            if "inline_data" in part:
                                image_data = part["inline_data"].get("data")
                                if image_data:
                                    image_bytes = base64.b64decode(image_data)
                                    print(f"  ✓ Gemini {variant.value} generated")
                                    return image_bytes

                logger.error("No image data in Gemini response")
                return None
            else:
                error_text = await response.text()
                logger.error(f"Gemini generation failed: {error_text}")
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as e:
        logger.error(f"Gemini generation error: {e}")
        return None


# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load font at specified size (cross-platform)

    Tries multiple font paths for macOS, Linux, and Windows.
    Falls back to default font if none are found.

    Args:
        size: Font size in points

    Returns:
        ImageFont.FreeTypeFont instance
    """
    # Try each font path in order
    for font_path in FONT_PATHS:
        try:
            # Try bold variant first (index 1 for TTC files)
            if font_path.endswith('.ttc'):
                font = ImageFont.truetype(font_path, size, index=1)
                return font
            else:
                # For TTF files, just load normally
                font = ImageFont.truetype(font_path, size)
                return font
        except (OSError, IOError):
            continue

    # If no fonts found, use default
    logger.warning("Could not load any system fonts, using default")
    return ImageFont.load_default()


def add_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    text: str,
    position: Tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill_color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 3
):
    """Draw text with outline stroke for visibility on any background"""
    x, y = position

    # Draw stroke by drawing text in all directions
    for offset_x in range(-stroke_width, stroke_width + 1):
        for offset_y in range(-stroke_width, stroke_width + 1):
            if offset_x != 0 or offset_y != 0:
                draw.text(
                    (x + offset_x, y + offset_y),
                    text,
                    font=font,
                    fill=stroke_color
                )

    # Draw main text on top
    draw.text((x, y), text, font=font, fill=fill_color)


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    draw: ImageDraw.ImageDraw
) -> List[str]:
    """Wrap text to fit within max_width"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def add_logo_with_shadow(
    base_img: Image.Image,
    logo_path: Path,
    position: Tuple[int, int],
    size: Tuple[int, int]
) -> Image.Image:
    """Composite logo with drop shadow onto base image"""

    # Load and resize logo
    logo = Image.open(logo_path).convert('RGBA')
    logo = logo.resize(size, Image.Resampling.LANCZOS)

    # Create shadow layer
    shadow = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
    shadow_mask = Image.new('L', logo.size, 0)
    shadow_draw = ImageDraw.Draw(shadow_mask)
    shadow_draw.rectangle([(0, 0), logo.size], fill=80)  # 30% opacity

    # Paste shadow with offset
    shadow_pos = (position[0] + 5, position[1] + 5)
    shadow.paste(logo, shadow_pos, shadow_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))

    # Composite shadow onto base
    base_img = Image.alpha_composite(base_img.convert('RGBA'), shadow)

    # Composite logo onto base
    base_img.paste(logo, position, logo)

    return base_img


def process_and_save(
    image_bytes: bytes,
    episode_num: int,
    episode_title: str,
    variant: ImageVariant,
    output_path: Path
) -> bool:
    """Main post-processing pipeline: load → overlay bar → grouped title → logo → save"""

    try:
        # Load image from bytes
        img = Image.open(BytesIO(image_bytes))

        # Ensure RGBA mode for compositing
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Handle resizing based on variant
        target_size = SQUARE_SIZE if variant == ImageVariant.SQUARE else SOCIAL_SIZE

        if img.size != target_size:
            # For social format with square source (OpenAI 1024x1024), use smart aspect ratio handling
            if variant == ImageVariant.SOCIAL and img.size[0] == img.size[1]:
                # Source is square, target is landscape - avoid distortion
                # Scale to fit height, then create canvas with extended/infilled background
                target_width, target_height = target_size

                # Scale image to fit height
                scale_factor = target_height / img.size[1]
                scaled_width = int(img.size[0] * scale_factor)
                scaled_img = img.resize((scaled_width, target_height), Image.Resampling.LANCZOS)

                # Create canvas
                canvas = Image.new('RGBA', target_size, (0, 0, 0, 255))

                # Create infilled background by stretching and heavily blurring the image edges
                bg = img.copy()
                # Stretch to full width to fill letterbox areas
                bg = bg.resize(target_size, Image.Resampling.LANCZOS)
                # Heavy blur to create seamless infill effect
                bg = bg.filter(ImageFilter.GaussianBlur(50))

                # Darken and desaturate slightly for better text contrast
                from PIL import ImageEnhance
                bg = ImageEnhance.Brightness(bg).enhance(0.5)  # Darken to 50%
                bg = ImageEnhance.Color(bg).enhance(0.7)       # Desaturate slightly

                # Composite: blurred infill background + scaled centered image
                canvas = Image.alpha_composite(canvas, bg)

                # Center the properly-scaled image
                x_offset = (target_width - scaled_width) // 2
                canvas.paste(scaled_img, (x_offset, 0), scaled_img)

                img = canvas
            else:
                # Normal resize for other cases
                img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Create overlay bar at bottom for title/logo
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        if variant == ImageVariant.SQUARE:
            # Square format: bottom bar with grouped title
            bar_height = SQUARE_BAR_HEIGHT
            bar_y = img.size[1] - bar_height

            # Draw semi-transparent black bar at bottom
            overlay_draw.rectangle(
                [(0, bar_y), (img.size[0], img.size[1])],
                fill=(0, 0, 0, TITLE_BAR_ALPHA)
            )

            # Composite overlay onto image
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

            # Episode number and title - grouped together, centered
            episode_font = get_font(SQUARE_EPISODE_FONT_SIZE)
            title_font = get_font(SQUARE_TITLE_FONT_SIZE)

            episode_text = f"Episode {episode_num}"

            # Wrap title to fit
            max_title_width = img.size[0] - (2 * TITLE_BAR_PADDING)
            title_lines = wrap_text(episode_title, title_font, max_title_width, draw)

            # Calculate total content height
            episode_bbox = draw.textbbox((0, 0), episode_text, font=episode_font)
            episode_height = episode_bbox[3] - episode_bbox[1]

            title_line_height = SQUARE_TITLE_FONT_SIZE + SQUARE_LINE_SPACING
            total_content_height = episode_height + SQUARE_EPISODE_TITLE_GAP + (len(title_lines) * title_line_height)

            # Center vertically in the bar
            content_start_y = bar_y + (bar_height - total_content_height) // 2

            # Draw episode number (centered)
            episode_width = episode_bbox[2] - episode_bbox[0]
            episode_x = (img.size[0] - episode_width) // 2
            draw.text((episode_x, content_start_y), episode_text, font=episode_font, fill="white")

            # Draw title lines (centered, below episode)
            title_y = content_start_y + episode_height + SQUARE_EPISODE_TITLE_GAP
            for line in title_lines:
                line_bbox = draw.textbbox((0, 0), line, font=title_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (img.size[0] - line_width) // 2
                draw.text((line_x, title_y), line, font=title_font, fill="white")
                title_y += title_line_height

            # Add logo in the bar (bottom-right corner)
            logo = Image.open(LOGO_PATH).convert('RGBA')
            logo = logo.resize(SQUARE_LOGO_SIZE, Image.Resampling.LANCZOS)
            logo_x = img.size[0] - SQUARE_LOGO_SIZE[0] - TITLE_BAR_PADDING
            logo_y = img.size[1] - SQUARE_LOGO_SIZE[1] - TITLE_BAR_PADDING
            img.paste(logo, (logo_x, logo_y), logo)

        else:  # SOCIAL format
            # Social format: bottom bar with grouped title (left-aligned)
            bar_height = SOCIAL_BAR_HEIGHT
            bar_y = img.size[1] - bar_height

            # Draw semi-transparent black bar at bottom
            overlay_draw.rectangle(
                [(0, bar_y), (img.size[0], img.size[1])],
                fill=(0, 0, 0, TITLE_BAR_ALPHA)
            )

            # Composite overlay onto image
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

            # Episode number and title - grouped together, left-aligned
            episode_font = get_font(SOCIAL_EPISODE_FONT_SIZE)
            title_font = get_font(SOCIAL_TITLE_FONT_SIZE)

            episode_text = f"Episode {episode_num}"

            # Calculate available width (accounting for logo on right)
            available_width = img.size[0] - SOCIAL_LOGO_SIZE[0] - (3 * TITLE_BAR_PADDING)
            title_lines = wrap_text(episode_title, title_font, available_width, draw)

            # Calculate positioning
            episode_bbox = draw.textbbox((0, 0), episode_text, font=episode_font)
            episode_height = episode_bbox[3] - episode_bbox[1]

            title_line_height = SOCIAL_TITLE_FONT_SIZE + SOCIAL_LINE_SPACING

            # Center vertically in the bar
            total_content_height = episode_height + SOCIAL_EPISODE_TITLE_GAP + (len(title_lines) * title_line_height)
            content_start_y = bar_y + (bar_height - total_content_height) // 2

            # Draw episode number
            text_x = TITLE_BAR_PADDING
            draw.text((text_x, content_start_y), episode_text, font=episode_font, fill="white")

            # Draw title lines (below episode)
            title_y = content_start_y + episode_height + SOCIAL_EPISODE_TITLE_GAP
            for line in title_lines:
                draw.text((text_x, title_y), line, font=title_font, fill="white")
                title_y += title_line_height

            # Add logo in the bar (bottom-right corner)
            logo = Image.open(LOGO_PATH).convert('RGBA')
            logo = logo.resize(SOCIAL_LOGO_SIZE, Image.Resampling.LANCZOS)
            logo_x = img.size[0] - SOCIAL_LOGO_SIZE[0] - TITLE_BAR_PADDING
            logo_y = img.size[1] - SOCIAL_LOGO_SIZE[1] - TITLE_BAR_PADDING
            img.paste(logo, (logo_x, logo_y), logo)

        # Convert to RGB for JPEG
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # Use alpha as mask
            img = rgb_img

        # Atomic write: save to temporary file first, then rename
        import tempfile
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temp file in same directory to ensure same filesystem
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            dir=output_path.parent,
            prefix=output_path.stem + '_'
        )

        try:
            # Close the file descriptor, we'll use the path
            os.close(temp_fd)

            # Save to temporary file
            img.save(temp_path, 'JPEG', quality=95, optimize=True)

            # Atomic rename (POSIX guarantees this is atomic)
            os.replace(temp_path, output_path)

            print(f"  ✓ Saved: {output_path}")
            return True
        except Exception as e:
            # Clean up temp file if something went wrong
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            raise e

    except (OSError, IOError, ValueError) as e:
        logger.error(f"Post-processing failed: {e}")
        return False


# ============================================================================
# ORCHESTRATION AND WORKFLOW
# ============================================================================

async def generate_all_variants(
    session: aiohttp.ClientSession,
    api_key: str,
    provider: Provider,
    concept: str,
    episode_num: int,
    episode_title: str,
    output_dir: Path
) -> Dict[ImageVariant, List[Path]]:
    """Generate 2 different images for both square and social variants from one provider"""

    results = {ImageVariant.SQUARE: [], ImageVariant.SOCIAL: []}
    title_slug = slugify_title(episode_title)

    # Generate 2 different images from the same concept
    for variant_num in range(1, 3):  # 1 and 2
        print(f"  🎨 {provider.value.capitalize()} variant {variant_num}/2...")

        if provider == Provider.OPENAI:
            prompt = build_image_prompt(concept, ImageVariant.SQUARE, Provider.OPENAI)
            base_image_bytes = await retry_with_backoff(generate_image_openai, session, api_key, prompt, ImageVariant.SQUARE, concept)
        else:  # GEMINI
            prompt = build_image_prompt(concept, ImageVariant.SQUARE, Provider.GEMINI)
            base_image_bytes = await retry_with_backoff(generate_image_gemini, session, api_key, prompt, ImageVariant.SQUARE, concept)

        if base_image_bytes:
            # Save square variant
            filename = f"{episode_num}-{title_slug}-{provider.value}-{variant_num}.jpg"
            output_path = output_dir / filename
            success = process_and_save(base_image_bytes, episode_num, episode_title, ImageVariant.SQUARE, output_path)
            if success:
                results[ImageVariant.SQUARE].append(output_path)

            # Reuse same image for social variant (different cropping/processing)
            filename_social = f"{episode_num}-{title_slug}-social-{provider.value}-{variant_num}.jpg"
            output_path_social = output_dir / filename_social
            success_social = process_and_save(base_image_bytes, episode_num, episode_title, ImageVariant.SOCIAL, output_path_social)
            if success_social:
                results[ImageVariant.SOCIAL].append(output_path_social)
        else:
            print(f"  ⚠️  {provider.value.capitalize()} variant {variant_num} generation failed")

    return results


async def generate_with_providers(
    episode_num: int,
    episode_title: str,
    selected_concept: str,
    providers: List[Provider]
) -> Dict[Provider, Dict[ImageVariant, List[Path]]]:
    """Generate 2 images per provider (4 square + 4 social total)"""

    print(f"\n🖼️  Generating images...")
    print("=" * 70)

    all_results = {}

    # Use flat output directory
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        for provider in providers:
            print(f"\n📡 {provider.value.upper()} Generation:")
            print("-" * 70)

            if provider == Provider.OPENAI and not OPENAI_API_KEY:
                print("  ⚠️  OpenAI API key not found, skipping")
                continue

            if provider == Provider.GEMINI and not GOOGLE_API_KEY:
                print("  ⚠️  Google API key not found, skipping")
                continue

            api_key = OPENAI_API_KEY if provider == Provider.OPENAI else GOOGLE_API_KEY

            results = await generate_all_variants(
                session,
                api_key,
                provider,
                selected_concept,
                episode_num,
                episode_title,
                output_dir
            )

            all_results[provider] = results

    return all_results


def save_concepts(episode_num: int, episode_title: str, concepts: List[Tuple[str, str]], selected_index: int):
    """Save concepts to JSON file for reference (atomic write)"""
    import tempfile

    # Convert concepts to dict format for JSON
    concepts_list = [
        {"concept": concept, "provider": provider}
        for concept, provider in concepts
    ]

    selected_concept, selected_provider = concepts[selected_index - 1]

    concepts_data = {
        "episode": episode_num,
        "title": episode_title,
        "concepts": concepts_list,
        "selected_index": selected_index,
        "selected_concept": selected_concept,
        "selected_provider": selected_provider
    }

    # Save to flat output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    concepts_file = OUTPUT_DIR / f"{episode_num}-concepts.json"

    # Atomic write: write to temp file first, then rename
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        dir=OUTPUT_DIR,
        prefix=f"{episode_num}-concepts_"
    )

    try:
        # Write JSON to temp file
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(concepts_data, f, indent=2)

        # Atomic rename
        os.replace(temp_path, concepts_file)

        print(f"  ✓ Concepts saved to: {concepts_file}")
    except Exception as e:
        # Clean up temp file if something went wrong
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        logger.error(f"Failed to save concepts: {e}")
        raise e


def print_summary(
    episode_num: int,
    episode_title: str,
    selected_concept: str,
    results: Dict[Provider, Dict[ImageVariant, List[Path]]]
):
    """Print generation summary"""

    print("\n")
    print("=" * 70)
    print("🎉 GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nEpisode {episode_num}: {episode_title}")
    print(f"Concept: {selected_concept}")
    print("\nGenerated images:")

    for provider, variants in results.items():
        print(f"\n{provider.value.upper()}:")
        for variant, paths in variants.items():
            size = "3000×3000" if variant == ImageVariant.SQUARE else "1200×630"
            if paths and len(paths) > 0:
                for i, path in enumerate(paths, 1):
                    print(f"  ✓ Variant {i}: {path} ({size})")
            else:
                print(f"  ✗ {variant.value} generation failed")

    print("\n" + "=" * 70)


# ============================================================================
# CLI AND MAIN
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate podcast cover images for The Cloud Pod"
    )
    parser.add_argument(
        '--episode',
        type=int,
        required=False,
        help='Episode number (e.g., 336)'
    )
    parser.add_argument(
        '--title',
        type=str,
        required=False,
        help='Episode title (e.g., "We Were Right (Mostly), 2026")'
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'gemini', 'both'],
        default='both',
        help='AI provider to use (default: both)'
    )
    parser.add_argument(
        '--skip-concepts',
        action='store_true',
        help='Skip concept generation (for testing)'
    )
    parser.add_argument(
        '--concept',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Pre-select concept number (1-6) (for testing)'
    )

    return parser.parse_args()


def get_interactive_input(args):
    """Get missing arguments interactively with validation"""

    # Episode number
    if args.episode is None:
        while True:
            try:
                episode_input = input("\nEpisode number: ").strip()
                if not episode_input:
                    print("❌ Episode number is required")
                    continue
                episode = int(episode_input)
                if validate_episode_number(episode):
                    args.episode = episode
                    break
                else:
                    print(f"❌ Episode number must be between {MIN_EPISODE} and {MAX_EPISODE}")
            except ValueError:
                print("❌ Please enter a valid number")

    # Episode title
    if args.title is None:
        while True:
            title_input = input("Episode title: ").strip()
            if not title_input:
                print("❌ Episode title is required")
                continue
            if validate_title(title_input):
                args.title = title_input
                break
            else:
                print(f"❌ Title must be 1-{MAX_TITLE_LENGTH} characters")

    return args


async def main():
    """Main execution function"""

    args = parse_args()

    # Get missing arguments interactively
    args = get_interactive_input(args)

    # Validate inputs (in case provided via CLI without interactive prompt)
    if not validate_episode_number(args.episode):
        print(f"❌ Error: Episode number must be between {MIN_EPISODE} and {MAX_EPISODE}")
        sys.exit(1)

    if not validate_title(args.title):
        print(f"❌ Error: Title must be 1-{MAX_TITLE_LENGTH} characters and not empty")
        sys.exit(1)

    # Validate API keys
    if not OPENAI_API_KEY and not GOOGLE_API_KEY:
        print("❌ Error: No API keys found!")
        print("Please set OPENAI_API_KEY and/or GOOGLE_API_KEY in .env file")
        sys.exit(1)

    # Validate logo file
    if not LOGO_PATH.exists():
        print(f"❌ Error: Logo file not found at {LOGO_PATH}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🎙️  THE CLOUD POD - Cover Image Generator")
    print("=" * 70)
    print(f"\nEpisode: {args.episode}")
    print(f"Title: {args.title}")
    print(f"Provider: {args.provider}")

    # Phase 1: Generate concepts (with regeneration loop)
    if args.skip_concepts:
        # Create dummy concepts for testing
        concepts = [
            (f"Abstract visualization of {args.title} using cloud computing iconography", "Test"),
            (f"Geometric representation of {args.title} in modern tech aesthetic", "Test"),
            (f"Metaphorical illustration of {args.title} with cloud elements", "Test"),
            (f"Digital cloud infrastructure concept for {args.title}", "Test"),
            (f"Futuristic tech visualization of {args.title}", "Test"),
            (f"Modern cloud metaphor for {args.title}", "Test")
        ]
        selected_index = args.concept or 1
        selected_concept, _ = concepts[selected_index - 1]
        print(f"\n⚠️  Skipping concept generation, using test concept {selected_index}")
    else:
        # Loop for concept generation/regeneration
        while True:
            concepts = await generate_concepts(args.title)

            # Check if concept generation failed
            if not concepts:
                print("\n❌ Concept generation failed. Exiting.")
                sys.exit(1)

            # If concept is pre-selected via CLI, use it (no interaction)
            if args.concept:
                selected_index = args.concept
                selected_concept, _ = concepts[selected_index - 1]
                print(f"\n✓ Auto-selected concept {selected_index}: {selected_concept}\n")
                break

            # Present concepts and get user choice
            selected_index, selected_concept, should_regenerate = await present_concepts_and_choose(concepts, args.title)

            # If user wants to regenerate, loop again
            if should_regenerate:
                continue
            else:
                break  # User selected a concept, exit loop

    # Save concepts
    save_concepts(args.episode, args.title, concepts, selected_index)

    # Phase 2 & 3: Generate and process images
    # Process Gemini first (faster), then OpenAI
    providers = []
    if args.provider == 'both':
        if GOOGLE_API_KEY:
            providers.append(Provider.GEMINI)
        if OPENAI_API_KEY:
            providers.append(Provider.OPENAI)
    elif args.provider == 'openai' and OPENAI_API_KEY:
        providers.append(Provider.OPENAI)
    elif args.provider == 'gemini' and GOOGLE_API_KEY:
        providers.append(Provider.GEMINI)

    if not providers:
        print("❌ No valid providers available with API keys")
        sys.exit(1)

    results = await generate_with_providers(
        args.episode,
        args.title,
        selected_concept,
        providers
    )

    # Print summary
    print_summary(args.episode, args.title, selected_concept, results)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
        sys.exit(0)
