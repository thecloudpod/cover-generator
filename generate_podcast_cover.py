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

# OpenAI Endpoints - Latest Models
OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_IMAGE_ENDPOINT = "https://api.openai.com/v1/images/generations"
OPENAI_CHAT_MODEL = "gpt-5.2"  # Latest reasoning model with thinking
OPENAI_IMAGE_MODEL = "gpt-image-1.5"  # Latest DALL-E with 5-reference support

# Gemini Endpoints - Gemini 3 Models
GEMINI_TEXT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
GEMINI_IMAGE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
GEMINI_TEXT_MODEL = "gemini-3-flash-preview"  # Latest flash with thinking_level
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"  # Nano Banana Pro - 4K with up to 14 references

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


def load_host_references() -> List[Dict[str, str]]:
    """Load all host images with metadata for 5-reference strategy

    Returns list of dicts with 'name', 'data' (base64), 'role', 'priority'
    Ordered by priority for reference hierarchy (Gemini 3 / OpenAI multi-reference)
    """
    hosts_dir = SCRIPT_DIR / "Hosts"

    # Define reference hierarchy - priority order matters for identity locking
    reference_order = [
        {"filename": "bolt.png", "name": "Bolt", "role": "Primary mascot reference (canonical)", "priority": 1},
        {"filename": "Jonathan Baker.png", "name": "Jonathan", "role": "Host 1 - Face and proportions reference", "priority": 2},
        {"filename": "Justin Brodley.jpg", "name": "Justin", "role": "Host 2 - Face and proportions reference", "priority": 3},
        {"filename": "Matthew Kohn.jpeg", "name": "Matthew", "role": "Host 3 - Face and proportions reference", "priority": 4},
        {"filename": "Ryan Lucas.jpeg", "name": "Ryan", "role": "Host 4 - Face and proportions reference", "priority": 5},
    ]

    loaded_refs = []

    try:
        for ref_spec in reference_order:
            ref_path = hosts_dir / ref_spec["filename"]

            if not ref_path.exists():
                logger.warning(f"Reference image not found: {ref_spec['filename']}")
                continue

            # Validate file size
            file_size = ref_path.stat().st_size
            if file_size > MAX_IMAGE_SIZE:
                logger.warning(f"{ref_spec['filename']} too large ({file_size / 1024 / 1024:.1f}MB), skipping")
                continue

            with open(ref_path, 'rb') as f:
                image_data = f.read()
                loaded_refs.append({
                    "name": ref_spec["name"],
                    "data": base64.b64encode(image_data).decode('utf-8'),
                    "role": ref_spec["role"],
                    "priority": ref_spec["priority"],
                    "filename": ref_spec["filename"]
                })

        if loaded_refs:
            print(f"  ✓ Loaded {len(loaded_refs)} reference images in priority order:")
            for ref in loaded_refs:
                print(f"    {ref['priority']}. {ref['name']} - {ref['role']}")

        return loaded_refs
    except (OSError, IOError) as e:
        logger.warning(f"Could not load reference images: {e}")
        return []


def build_reference_hierarchy_instructions(references: List[Dict[str, str]], include_bolt: bool, include_hosts: bool) -> str:
    """Build PIXEL PRIORITY MODE instructions for multi-reference image generation"""
    if not references or (not include_bolt and not include_hosts):
        return ""

    # Filter references based on what's included
    active_refs = [ref for ref in references
                   if (ref["name"] == "Bolt" and include_bolt) or (ref["name"] != "Bolt" and include_hosts)]

    ref_list = "\n".join(f"Image {ref['priority']}: {ref['name']} - {ref['role']}" for ref in active_refs)

    sections = [
        "\n\n🔒 PIXEL PRIORITY MODE - IDENTITY LOCK: ABSOLUTE\n",
        "REFERENCE IMAGE HIERARCHY (in priority order):\n",
        ref_list,
        "\n\n⚠️ CRITICAL INSTRUCTIONS - IDENTITY LOCK PROTOCOL:\n"
    ]

    if include_bolt:
        sections.append("""
• BOLT (Image 1): This is the CANONICAL SOURCE OF TRUTH for Bolt the mascot
  - Match EXACT colors (#0066FF blue body, yellow lightning bolt)
  - Match EXACT proportions and cloud-form body shape
  - Match EXACT headphones, antenna, and facial features
  - DO NOT average or reinterpret - copy this design exactly
""")

    if include_hosts:
        sections.append("""
• HOSTS (Images 2-5): These are the SOURCE OF TRUTH for the four podcast hosts
  - Use the composite visual data from all host reference images
  - Match facial structure, features, hair patterns, and proportions from photos
  - CRITICAL: Render hosts in the SAME CARTOON STYLE as Bolt
  - Use simple geometric body shapes (rounded rectangles, circles) like Bolt's cloud form
  - Minimal facial features with personality (dots/simple shapes for eyes, simple curves for mouths) matching Bolt's style
  - Clean flat vector aesthetic matching Bolt's illustration style exactly
  - Same level of simplification and abstraction as Bolt - NOT photorealistic, NOT detailed portraits
  - Preserve distinctive features in cartoon form: bald head, horseshoe hair, facial hair patterns, face shapes
  - Each host should look like a "brother character" to Bolt in the same visual universe
  - DO NOT add people not shown in the references
""")

    sections.append("""
• RENDERING STRATEGY:
  - Modern flat illustration style (Kurzgesagt/Slack aesthetic)
  - Simplified geometric shapes while maintaining facial recognizability
  - Each character must be identifiable as the person in their reference image
  - Characters should look like illustrated versions of the real people, not generic characters

• AVOID:
  - Averaging facial features across references
  - Creating generic placeholder people
  - Reinterpreting or redesigning characters
  - Adding extra people beyond the references provided
""")

    return "".join(sections)


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

THE CLOUD POD UNIVERSE CONTEXT:
- This podcast covers cloud computing and tech industry news with humor and insight
- The tone is always comedic and affectionate, never mean-spirited

BRAND CHARACTER NOTES (only include if mentioned in the concept):
- **Oracle - "The Evil Empire"**: When Oracle appears, portray as the bumbling, incompetent villain empire (think Dark Helmet from Spaceballs). Over-the-top evil aesthetic with comically inept execution. Big helmets, dramatic capes, "evil empire" branding, but everything goes wrong in silly ways. They're trying SO HARD to be menacing but keep tripping over themselves. The joke: we don't take their cloud seriously, so they're portrayed as wannabe villains who can't get anything right.
- AWS: Dominant cloud leader, professional but can be playfully corporate
- Google Cloud: Innovation-focused, sometimes quirky
- Azure: Enterprise-focused Microsoft cloud
- Only include these brands if they are explicitly mentioned in the episode concept or title

ILLUSTRATION STYLE - Modern flat vector aesthetic:
- Visual style reference: Think Kurzgesagt, Slack marketing illustrations, or modern tech editorial art
- Bold shapes with clean edges and smooth gradients
- Simple geometric body shapes for characters (rectangles, rounded forms)
- Minimal facial features with maximum personality (dots for eyes, simple curves for mouths)
- Playful proportions and exaggerated scale for comedic effect
- Soft shadows and depth through color variation, not harsh lighting

CHARACTER DESIGN PRINCIPLES (when characters appear):
- Distinctive silhouettes - recognizable even in black shadow
- Simple geometric shapes for bodies (readable at any size)
- Consistent proportions - hosts are similar scale to each other
- Color-coded for quick identification (each character has signature color)
- Each character has signature pose/gesture that expresses personality
- Clean separation - no overlapping characters, clear breathing room

COLOR PALETTE:
- Primary: Blues (#0066FF), cloud theme colors
- Accent: White, light grays, strategic pops of color (yellow for Bolt's lightning bolt)
- Character color-coding: Jonathan=light blue, Justin=gray, Matthew=warm brown/orange, Ryan=teal
- Vibrant but professional - saturated blues, clean whites, purposeful color choices
- Use color to enhance humor and guide the eye

COMPOSITION & FRAMING:
- Primary focal point in center-to-upper area with clear visual hierarchy
- Embrace negative space - don't fill every corner
- Breathing room around characters and key objects
- Characters shown as distinct stylized figures with recognizable visual signatures
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

    # Build dynamic sections
    sections = ["""You are the creative director for The Cloud Pod, a tech podcast famous for visual wordplay and literal humor."""]

    # Detect if title mentions AI/agent/bot/robot - these should map to Bolt
    ai_keywords = ['ai', 'agent', 'bot', 'robot', 'artificial', 'intelligence', 'llm', 'model', 'chatbot']
    title_lower = episode_title.lower()
    mentions_ai = any(keyword in title_lower for keyword in ai_keywords)

    sections.append(f"""

Episode Title: '{episode_title}'

YOUR TASK: Create ONE completely NEW visual scene based ONLY on '{episode_title}'. Take the words in this title literally and create a specific, original scene.

APPROACH - How to interpret titles literally:
- Take individual WORDS literally (if title says 'wardrobe', show actual clothing; if it says 'layers', show a visual stack)
- Convert abstract concepts into PHYSICAL objects (if title mentions 'conversational', show someone literally talking/yelling)
- Find the VISUAL PUN in the title wording
- Create SPECIFIC details (not vague 'tech vibes' but concrete physical objects, readable text, tangible props)
- Exaggerate for HUMOR (impossible proportions, absurd scales, playful contradictions)

CONCEPT VARIETY:
- Balance between CHARACTER-FOCUSED concepts and OBJECT/ENVIRONMENT concepts
- Character-based concepts often work great for storytelling and humor
- Minimalist object/environment concepts (empty server rack, floating cloud, literal interpretation) can also be effective
- Choose the approach that best serves the specific visual pun

IMPORTANT: Do NOT reuse visual elements from other episodes. Each concept must be completely original based on THIS episode title.""")

    if previous_concepts:
        concepts_list = "\n".join(f"{i}. {c}" for i, c in enumerate(previous_concepts, 1))
        sections.append(f"\n\nPREVIOUS CONCEPTS ALREADY GENERATED (do NOT repeat these ideas):\n{concepts_list}\n\nYour concept must be COMPLETELY DIFFERENT from these.")

    if keywords and keywords.strip():
        sections.append(f"\n\nKEYWORD GUIDANCE: The user wants concepts that incorporate or emphasize these themes: {keywords.strip()}\nUse these keywords to steer your creative direction while still interpreting the episode title literally.")

    # Dynamic character guidance based on title content
    if mentions_ai:
        sections.append(f"""

CHARACTER GUIDANCE - IMPORTANT FOR THIS EPISODE:
This title mentions AI/agents/bots, which should naturally map to BOLT, our cloud robot mascot!

- **Bolt** - The Cloud Pod's blue cloud robot mascot - PERFECT for representing AI agents, bots, or robots in the title
  → When the title says "AI Agent" or similar, Bolt IS that agent
  → Example: If title mentions "AI Agent Can't Keep Its Mouth Shut" → Show Bolt with mouth taped shut, secrets escaping, etc.
  → Bolt works great as the main character for comedy, mishaps, and visual gags

- **The Four Hosts** - Jonathan, Justin, Matthew, Ryan - Use when you need human characters or a team dynamic

PRIORITY: Since this title mentions AI/agents, strongly consider using Bolt as the central character.""")
    else:
        sections.append(f"""

AVAILABLE CHARACTERS (use when they enhance the concept):
- **Bolt** - The Cloud Pod's blue cloud robot mascot - great for tech concepts, comedy, and character-driven visual gags
- **The Four Hosts** - Jonathan, Justin, Matthew, Ryan - use for human-focused concepts or team dynamics

Choose character-based or environment-based concepts based on what best serves the visual pun.""")

    sections.append(f"""

Return ONLY one sentence describing the specific visual scene based on '{episode_title}':""")

    return "".join(sections)


def build_image_prompt(concept: str, variant: ImageVariant, provider: Provider = None, include_bolt: bool = False, include_hosts: bool = False) -> str:
    """Build detailed prompt for image generation with optional model-specific emphasis

    Args:
        concept: The visual concept description
        variant: Square or social media format
        provider: OpenAI or Gemini (for model-specific emphasis)
        include_bolt: Whether to include Bolt character (user choice)
        include_hosts: Whether to include the four hosts (user choice)
    """

    format_type = "Square format (1:1 aspect ratio)." if variant == ImageVariant.SQUARE else "Horizontal landscape format (roughly 16:9 aspect ratio - WIDE not tall)."

    dimension_guidance = f"""{format_type}

⚠️ CRITICAL COMPOSITION FRAMING - READ CAREFULLY:
The lower 25% of this image will be COMPLETELY COVERED by a text overlay bar in post-production. ANY important visual elements placed in the bottom quarter will be HIDDEN and WASTED.

MANDATORY PLACEMENT RULES:
• ALL main subjects, characters, faces, key objects: Position in UPPER 75% ONLY (top and middle areas)
• Character faces: Must be in upper-middle to upper portion - NEVER in bottom quarter
• Important visual elements: Keep in top 3/4 of frame - pretend the bottom 25% doesn't exist for composition purposes
• Bottom quarter use ONLY: Simple environmental grounding (solid floors, horizon lines, gradient sky/background, atmospheric effects)
• Think of the canvas as 75% usable space for content + 25% reserved background space

COMPOSITION ANALOGY: Like a magazine cover where the bottom has the magazine title bar - all the interesting content stays ABOVE that bar.

Visual weight and focal points: Upper 75% of frame. Bottom 25%: Just background fill."""

    # Add Bolt guidance only if user selected it
    bolt_guidance = ""
    if include_bolt:
        bolt_guidance = """

BOLT - The Cloud Pod Mascot (reference image provided - match exactly):

CHARACTER DESIGN:
Body Shape: Puffy cloud form with rounded edges, roughly square proportions when at rest
Body Color: Bright electric blue (#0066FF) - solid, no gradients on main body
Chest Icon: Bold yellow lightning bolt (zigzag shape, pointing downward)
Face: Simple friendly features - dot eyes (dark blue), curved smile line
Accessories: Small rounded headphones on sides, single antenna on top with ball tip
Style: Clean flat vector shapes, matte finish, modern illustration aesthetic

PERSONALITY & POSE:
Expression: Always friendly and approachable, enthusiastic energy
Signature Gestures: Floating/bouncing, pointing with cloud appendages, excited poses
Scale: Roughly 60-70% the height of human hosts when appearing together
Consistency: Bolt's core design (blue color, lightning bolt orientation) stays identical across all scenes

RENDERING PRIORITY: Match the reference image design exactly - this character must be instantly recognizable across all episodes."""

    # Add hosts guidance only if user selected it
    hosts_guidance = ""
    if include_hosts:
        hosts_guidance = """

THE FOUR CLOUD POD HOSTS - Character Design Consistency Guide

STANDARD SCENARIO: The podcast has 4 core hosts - Jonathan, Justin, Matthew, and Ryan
- For typical scenes, render these 4 distinct individuals with their signature characteristics
- Each host has recognizable visual signatures (hair, facial hair, outfit colors, build)

CROWD/DUPLICATION SCENARIOS: When the concept requires many people (org charts, crowds, teams, armies, etc.)
- You may duplicate the hosts to fill the scene
- CRITICAL: Add distinguishing variations to each duplicate so they don't look identical:
  - Accessories: glasses, hats, headphones, scarves, capes, ties, badges
  - Props: laptops, phones, tennis rackets, canes, coffee cups, tablets
  - Costume variations: different shirt colors/styles, jackets, hoodies, formal wear
  - Hairstyle tweaks: different lengths, styles, colors (while maintaining base recognition)
  - Posture variations: sitting, standing, leaning, different arm positions
- Maintain core recognizability (Jonathan's build, Justin's bald head, Matthew's horseshoe, Ryan's wavy hair) but add visual variety
- Think: "alternate universe versions" or "the same person in different outfits/contexts"
- Examples:
  - Org chart: Same bald Justin at different levels - one in suit with glasses, one in hoodie with coffee, one in blazer with tablet
  - Crowd scene: Multiple Matthews - one with sunglasses, one with cap, one with scarf, all maintaining horseshoe hair and beard
  - Team meeting: Duplicated Ryans - different colored sweaters (teal, navy, gray), varied gestures, one with laptop

Illustration Style: Modern flat vector (Kurzgesagt/Slack aesthetic)
Key Principle: DISTINCTIVE SILHOUETTES + COLOR CODING + VISUAL SIGNATURES

SHARED DESIGN LANGUAGE:
- Simple geometric body shapes (rounded rectangles for torsos)
- Minimal facial features with personality (simple eyes, expressive mouths)
- Tech-casual professional attire (each has signature outfit)
- Similar scale to each other, all standing roughly same height
- Light skin tone, simple flat color rendering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHARACTER 1 - JONATHAN "The Welcoming Lead"
HAIR: Dark brown, slightly wavy, medium length with side part (full head of hair)
FACE: Clean-shaven smooth face, round friendly features, warm smile
BUILD: Fuller/broad rectangular body shape, solid presence (tallest overall impression)
OUTFIT: Light blue button-down shirt (sleeves rolled to elbows), khaki pants
SIGNATURE COLOR: Light blue (#5B9BD5) - appears in shirt and accents
SIGNATURE POSE: Upright posture, welcoming hand gestures (waving, pointing, presenting)
PERSONALITY: Open body language, leading the group, friendly approachable energy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHARACTER 2 - JUSTIN "The Distinguished Sage"
HAIR: Completely bald head (smooth dome, subtle highlights for dimension)
FACIAL HAIR: Silver-gray goatee (covers chin and upper lip only, neat and trimmed)
BUILD: Robust/solid rectangular shape, broad shoulders, strong presence
OUTFIT: Gray blazer over black t-shirt, dark jeans, smart-casual
SIGNATURE COLOR: Gray (#808080) - appears in blazer and overall palette
SIGNATURE POSE: Thoughtful stance - hand on chin, crossed arms, grounded stable posture
PERSONALITY: Wise mentor energy, contemplative, authoritative but approachable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHARACTER 3 - MATTHEW "The Energetic Optimist"
HAIR: Horseshoe pattern - bald shiny dome on top, brown hair wraps around sides and back from ear to ear (render as "C" shape when viewed from front)
FACIAL HAIR: Full scruffy brown beard (covers cheeks, chin, jawline, connects to side hair)
BUILD: Slimmer/athletic rectangular shape, more vertical proportions, lean
OUTFIT: Warm brown or orange polo shirt with rolled sleeves, casual pants
SIGNATURE COLOR: Warm brown/orange (#D97B3E) - appears in shirt
SIGNATURE POSE: Dynamic animated gestures, mid-motion, energetic pointing or jumping
PERSONALITY: Always smiling, enthusiastic energy, most animated of the group

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHARACTER 4 - RYAN "The Creative Thinker"
HAIR: Shortish wavy golden-brown hair with natural volume and texture (full head, tousled)
FACIAL HAIR: Darker brown goatee (covers chin and upper lip, similar coverage to Justin but brown)
BUILD: Medium balanced rectangular shape, average proportions
OUTFIT: Teal/turquoise crew-neck sweater or henley, modern casual
SIGNATURE COLOR: Teal (#4AAAA5) - appears in shirt/sweater
SIGNATURE POSE: Slightly angled stance, thoughtful gestures (looking at objects, contemplating)
PERSONALITY: Creative contemplation, often interacting with tech objects or screens

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VISUAL IDENTIFICATION SYSTEM:
Primary ID: Hair pattern (Justin=bald, Matthew=horseshoe/bald top, Jonathan=full dark, Ryan=wavy golden-brown)
Secondary ID: Facial hair (Matthew=full beard, Justin=gray goatee, Ryan=brown goatee, Jonathan=clean)
Tertiary ID: Signature color (blue, gray, orange, teal)
Quaternary ID: Build (Jonathan=broad, Justin=solid, Matthew=lean, Ryan=balanced)

GROUP COMPOSITION (when all 4 appear together):
- Standard order left-to-right: Jonathan → Justin → Matthew → Ryan
- Spacing: Clear breathing room between each character, no overlap
- Scale: All roughly similar height, similar size to each other
- Arrangement: Arc, line, or informal cluster (avoid rigid lineup unless concept calls for it)
- Interaction: Characters react to scene elements or look at shared focal point
- With Bolt: Mascot appears in center or floating above, roughly 60-70% of host height

CONSISTENCY CHECKLIST (for standard 4-host scenes):
✓ Can you identify each host by silhouette alone?
✓ Is each character's signature color visible in their outfit?
✓ Are the four hair patterns clearly distinct? (full, bald, horseshoe, wavy)
✓ Do facial hair patterns match? (none, gray goatee, full beard, brown goatee)
✓ Are they in recognizable left-to-right order when in group?
✓ Does each pose match their personality archetype?

DUPLICATION CHECKLIST (for crowd/org chart scenes):
✓ If hosts are duplicated, does each copy have distinguishing features?
✓ Are accessories/props varied across duplicates (glasses, hats, different colored shirts)?
✓ Can you still recognize the "base" host despite variations (bald head, horseshoe hair, etc.)?
✓ Do variations feel natural and not random?

RENDERING PRIORITY: Core hosts must be instantly recognizable through their visual signatures, even when varied for crowd scenes."""

    # Model-specific style emphasis
    model_emphasis = ""
    if provider == Provider.OPENAI:
        model_emphasis = "\n\nOPENAI MODEL OPTIMIZATION:\n• Prioritize clean flat illustration with bold shapes and smooth color gradients\n• Leverage strong facial expression capabilities - show personality through simple expressive faces\n• Use lighting and depth to separate characters while maintaining flat aesthetic\n• Consistent proportions are key - maintain character scale relationships\n• Avoid photorealistic rendering - stay in illustration style"
    elif provider == Provider.GEMINI:
        # Add duplication variation reminder for Gemini
        duplication_reminder = ""
        if include_hosts:
            duplication_reminder = "\n• If duplicating hosts for crowd scenes, add accessories/costume variations to each copy (glasses, different colored shirts, props, hats, etc.)"

        model_emphasis = f"\n\nGEMINI MODEL OPTIMIZATION:\n• Match exact visual style and character designs from reference images provided\n• Emphasize geometric shape language - clean simple forms\n• Strong on color consistency - use signature colors to distinguish characters\n• Test silhouettes - characters should be recognizable in black shadow\n• Maintain consistent character design across entire scene{duplication_reminder}"

    # Build mandatory character inclusion instruction
    character_requirement = ""
    if include_hosts and include_bolt:
        character_requirement = """

🎭 MANDATORY CHARACTER INCLUSION - THIS VARIANT MUST INCLUDE:
• Bolt (the blue cloud robot mascot) - REQUIRED in this scene
• All four podcast hosts (Jonathan, Justin, Matthew, Ryan) - REQUIRED in this scene

These characters MUST appear in this image, integrated naturally into the concept described above.
Arrange them as a group interacting with the scene elements, reacting to the situation, or participating in the visual story.
Do not skip any characters - all five must be present and clearly visible."""
    elif include_bolt:
        character_requirement = """

🎭 MANDATORY CHARACTER INCLUSION - THIS VARIANT MUST INCLUDE:
• Bolt (the blue cloud robot mascot) - REQUIRED in this scene

Bolt MUST appear in this image, integrated naturally into the concept."""

    return f"""{BASE_STYLE_PROMPT}

SPECIFIC EPISODE CONCEPT:
{concept}

CRITICAL: Render the elements described in the concept above. Do not add unrelated objects or text that are not explicitly mentioned in the concept.{character_requirement}
{bolt_guidance}
{hosts_guidance}

COMPOSITION REQUIREMENTS:
{dimension_guidance}

Generate a background image that visualizes this concept while maintaining The Cloud Pod's professional tech aesthetic.{model_emphasis}"""


# ============================================================================
# TEXT GENERATION FUNCTIONS (Concept Phase)
# ============================================================================

async def _make_api_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
    provider_name: str,
    extract_response
) -> Optional[str]:
    """Generic API request handler for both OpenAI and Gemini"""
    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return extract_response(data)
            else:
                error_text = await response.text()
                logger.error(f"{provider_name} request failed: {error_text}")
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"{provider_name} request error: {e}")
        return None


async def generate_concept_openai(
    session: aiohttp.ClientSession,
    api_key: str,
    episode_title: str,
    previous_concepts: List[str] = None,
    keywords: str = None
) -> Optional[str]:
    """Generate a creative concept using OpenAI GPT-4"""
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [{"role": "user", "content": build_concept_prompt(episode_title, previous_concepts, keywords)}],
        "max_completion_tokens": 2000,
        "reasoning_effort": "medium"
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    return await _make_api_request(
        session, OPENAI_CHAT_ENDPOINT, payload, headers, "OpenAI",
        lambda data: data["choices"][0]["message"]["content"].strip()
    )


async def generate_concept_gemini(
    session: aiohttp.ClientSession,
    api_key: str,
    episode_title: str,
    previous_concepts: List[str] = None,
    keywords: str = None
) -> Optional[str]:
    """Generate a creative concept using Google Gemini"""
    payload = {
        "contents": [{"parts": [{"text": build_concept_prompt(episode_title, previous_concepts, keywords)}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 2048
        }
    }

    return await _make_api_request(
        session, f"{GEMINI_TEXT_ENDPOINT}?key={api_key}", payload, {}, "Gemini",
        lambda data: data["candidates"][0]["content"]["parts"][0]["text"].strip()
    )


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
                        "model": OPENAI_CHAT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": CONCEPT_REFINEMENT_TOKENS
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


async def polish_custom_concept(custom_concept: str, episode_title: str) -> Optional[str]:
    """Polish and improve a user's custom concept while preserving their intent"""
    prompt = f"""You are helping polish a custom visual concept for The Cloud Pod podcast cover.

Episode Title: "{episode_title}"
User's Custom Concept: "{custom_concept}"

YOUR TASK: Clean up and improve this concept while PRESERVING the user's core idea and intent.

IMPROVEMENTS TO MAKE:
- Add specific visual details (colors, scales, textures, spatial relationships)
- Enhance comedic elements or visual puns already present
- Ensure the concept fits The Cloud Pod's playful, professional tech aesthetic
- Add clarity about positioning, framing, and composition if vague
- If the concept mentions "Bolt" or characters, maintain that focus
- Keep the description to 1-2 sentences maximum

IMPORTANT: Keep the user's core idea intact. Only enhance and clarify, don't change the fundamental concept.

Return ONLY the polished concept:"""

    try:
        # Use OpenAI if available, otherwise Gemini
        provider = "OpenAI" if OPENAI_API_KEY else "Gemini"

        async with aiohttp.ClientSession() as session:
            if provider == "OpenAI" and OPENAI_API_KEY:
                async with session.post(
                    OPENAI_CHAT_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": OPENAI_CHAT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": CONCEPT_REFINEMENT_TOKENS
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

        logger.error("Custom concept polishing failed")
        return None
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as e:
        logger.error(f"Custom concept polishing error: {e}")
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
        print(f"Commands: [{concept_range}] = Select concept | W = Write your own concept | 0 = Generate 6 new concepts | M = Generate MORE concepts | {refine_range} = Refine concept | X = Exit")

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

                # Write your own concept
                if choice == 'W':
                    print("\n✍️  Write Your Own Concept")
                    print("-" * 70)
                    print("Describe the visual scene you want to create.")
                    print("Be specific about characters, objects, actions, and visual details.")
                    print("Example: 'Bolt with duct tape over his mouth, emerging from a room")
                    print("         filled with floating \"TOP SECRET\" folders and locked safes.'")
                    print("-" * 70)
                    custom_concept = input("\nYour concept: ").strip()

                    if custom_concept:
                        print(f"\n📝 Your concept: {custom_concept}")
                        print("\n⏳ Polishing your concept...")

                        # Polish the custom concept using AI
                        polished = await polish_custom_concept(custom_concept, episode_title)

                        if polished:
                            print(f"\n✨ Polished concept: {polished}\n")
                            confirm = input("Use this polished concept? (Y/n/e to edit): ").strip().lower()

                            if confirm == 'e':
                                # Allow user to edit the polished version
                                print("\nEdit the polished concept:")
                                edited_concept = input(f"{polished}\n> ").strip()
                                if edited_concept:
                                    concepts.append((edited_concept, "Custom"))
                                    return len(concepts), edited_concept, False
                                else:
                                    # User pressed enter without editing, use polished version
                                    concepts.append((polished, "Custom"))
                                    return len(concepts), polished, False
                            elif confirm != 'n':
                                # Add polished concept to list and return it
                                concepts.append((polished, "Custom"))
                                return len(concepts), polished, False
                            else:
                                # User rejected, try original
                                use_original = input("\nUse your original concept instead? (Y/n): ").strip().lower()
                                if use_original != 'n':
                                    concepts.append((custom_concept, "Custom"))
                                    return len(concepts), custom_concept, False
                        else:
                            # Polishing failed, offer to use original
                            print("\n⚠️  Polishing failed, but you can still use your original concept.")
                            confirm = input("Use your original concept? (Y/n): ").strip().lower()
                            if confirm != 'n':
                                concepts.append((custom_concept, "Custom"))
                                return len(concepts), custom_concept, False
                    else:
                        print("❌ Concept cannot be empty")
                        continue

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
                    print(f"Please enter 1-{len(concepts)}, W, 0, M, R#, or X")

            except ValueError:
                print(f"Please enter 1-{len(concepts)}, W, 0, M, R# (e.g., R3), or X")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                sys.exit(0)




# ============================================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================================

def _filter_references(references: List[Dict[str, str]], include_bolt: bool, include_hosts: bool) -> List[Dict[str, str]]:
    """Filter reference images based on what's requested"""
    return [ref for ref in references
            if (ref["name"] == "Bolt" and include_bolt) or (ref["name"] != "Bolt" and include_hosts)]


async def _extract_openai_image(session: aiohttp.ClientSession, data: dict, variant: ImageVariant) -> Optional[bytes]:
    """Extract image bytes from OpenAI API response"""
    if "data" in data and len(data["data"]) > 0:
        image_data = data["data"][0]
        if "b64_json" in image_data:
            print(f"  ✓ OpenAI {variant.value} generated")
            return base64.b64decode(image_data["b64_json"])
        elif "url" in image_data:
            async with session.get(image_data["url"]) as img_response:
                if img_response.status == 200:
                    print(f"  ✓ OpenAI {variant.value} generated")
                    return await img_response.read()
    return None


async def generate_image_openai(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    variant: ImageVariant,
    concept: str = "",
    include_bolt: bool = False,
    include_hosts: bool = False
) -> Optional[bytes]:
    """Generate image using OpenAI GPT Image 1.5 with 5-reference strategy"""

    print(f"  🎨 OpenAI GPT-Image-1.5 generating {variant.value} variant...")

    # Use edit endpoint if we have ANY reference images (Bolt and/or hosts)
    use_edit_endpoint = include_bolt or include_hosts

    if use_edit_endpoint:
        # Use images.edit endpoint with 5-reference strategy
        # gpt-image-1.5 supports up to 5 reference images
        import aiohttp

        form = aiohttp.FormData()
        form.add_field('model', OPENAI_IMAGE_MODEL)
        form.add_field('size', '1024x1024')
        form.add_field('quality', 'high')
        form.add_field('input_fidelity', 'high')  # Preserve details from input images
        form.add_field('output_format', 'png')
        form.add_field('n', '1')

        # Load and filter references
        references = load_host_references()
        filtered_refs = _filter_references(references, include_bolt, include_hosts)

        print(f"  📋 Loaded {len(references)} total references, filtered to {len(filtered_refs)} (Bolt={include_bolt}, Hosts={include_hosts})")

        if filtered_refs:
            for ref in filtered_refs:
                ref_bytes = base64.b64decode(ref["data"])
                content_type = "image/png" if ref["filename"].endswith('.png') else "image/jpeg"
                form.add_field('image[]', ref_bytes, filename=ref["filename"], content_type=content_type)

            print(f"  📸 Using {len(filtered_refs)} reference images with identity lock")

        # Build identity lock instructions and add to prompt
        reference_instructions = build_reference_hierarchy_instructions(references, include_bolt, include_hosts)
        enhanced_prompt = f"""{reference_instructions}

GENERATION TASK:
{prompt}"""

        form.add_field('prompt', enhanced_prompt)

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
                    result = await _extract_openai_image(session, data, variant)
                    if result:
                        return result
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
        if include_hosts:
            print("  📝 Using text descriptions for 4 hosts (Jonathan, Justin, Matthew, Ryan)")

        payload = {
            "model": OPENAI_IMAGE_MODEL,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "high",
            "output_format": "png"
        }

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            async with session.post(
                OPENAI_IMAGE_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=OPENAI_TIMEOUT)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result = await _extract_openai_image(session, data, variant)
                    if result:
                        return result
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
    concept: str = "",
    include_bolt: bool = False,
    include_hosts: bool = False
) -> Optional[bytes]:
    """Generate image using Google Gemini with optional reference images"""

    print(f"  🎨 Gemini 3 Pro Image (Nano Banana) generating {variant.value} variant...")

    # Load and filter reference images
    parts = []
    references = load_host_references()
    filtered_refs = _filter_references(references, include_bolt, include_hosts)

    print(f"  📋 Loaded {len(references)} total references, filtered to {len(filtered_refs)} (Bolt={include_bolt}, Hosts={include_hosts})")

    # Build prompt following Google's official SDK example:
    # Prompt FIRST, then reference images (no captions between)
    if filtered_refs:
        # Build concise prompt with character descriptions
        character_intro = ""
        if include_bolt:
            character_intro += "Bolt (the blue cloud robot mascot shown in the first reference image). "
        if include_hosts:
            character_intro += "The four podcast hosts (Jonathan, Justin, Matthew, Ryan) shown in the reference photos. Render them in the same cartoon style as Bolt - simple flat vector illustration with distinctive hair patterns: Jonathan has dark wavy hair and is clean-shaven, Justin is completely bald with gray goatee, Matthew has horseshoe hair pattern (bald on top, hair on sides) with full brown beard, Ryan has wavy golden-brown hair with brown goatee. Use their signature outfit colors: blue, gray, orange, teal."

        # Prompt comes FIRST (matching official SDK example)
        final_prompt = f"""An illustration featuring {character_intro}

GENERATION TASK:
{prompt}"""
        parts.append({"text": final_prompt})

        # Then add ALL reference images in sequence (no captions between)
        for ref in filtered_refs:
            mime_type = "image/jpeg" if ref["filename"].endswith(('.jpg', '.jpeg')) else "image/png"
            parts.append({"inline_data": {"mime_type": mime_type, "data": ref["data"]}})

    else:
        parts.append({"text": prompt})

    # Gemini image generation - matching official SDK config structure
    # Set aspect ratio and resolution based on variant
    aspect_ratio = "1:1" if variant == ImageVariant.SQUARE else "16:9"
    image_size = "4K"  # Maximum quality (options: "1K", "2K", "4K")

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size
            }
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

                # Extract image from Gemini response (try both camelCase and snake_case)
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            # Try both naming conventions
                            inline_data = part.get("inlineData") or part.get("inline_data")
                            if inline_data and inline_data.get("data"):
                                print(f"  ✓ Gemini {variant.value} generated")
                                return base64.b64decode(inline_data["data"])

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


def atomic_write(target_path: Path, write_func, *args, **kwargs):
    """Atomic file write using temp file + rename pattern"""
    import tempfile

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.tmp',
        dir=target_path.parent,
        prefix=target_path.stem + '_'
    )

    try:
        os.close(temp_fd)
        write_func(temp_path, *args, **kwargs)
        os.replace(temp_path, target_path)
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        raise


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

        # Get variant-specific parameters
        is_square = variant == ImageVariant.SQUARE
        bar_height = SQUARE_BAR_HEIGHT if is_square else SOCIAL_BAR_HEIGHT
        episode_font_size = SQUARE_EPISODE_FONT_SIZE if is_square else SOCIAL_EPISODE_FONT_SIZE
        title_font_size = SQUARE_TITLE_FONT_SIZE if is_square else SOCIAL_TITLE_FONT_SIZE
        line_spacing = SQUARE_LINE_SPACING if is_square else SOCIAL_LINE_SPACING
        episode_title_gap = SQUARE_EPISODE_TITLE_GAP if is_square else SOCIAL_EPISODE_TITLE_GAP
        logo_size = SQUARE_LOGO_SIZE if is_square else SOCIAL_LOGO_SIZE

        # Create overlay bar at bottom
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        bar_y = img.size[1] - bar_height
        overlay_draw.rectangle([(0, bar_y), (img.size[0], img.size[1])], fill=(0, 0, 0, TITLE_BAR_ALPHA))

        # Composite overlay onto image
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # Setup fonts and text
        episode_font = get_font(episode_font_size)
        title_font = get_font(title_font_size)
        episode_text = f"Episode {episode_num}"

        # Calculate available width for title wrapping
        max_title_width = img.size[0] - (2 * TITLE_BAR_PADDING) if is_square else img.size[0] - logo_size[0] - (3 * TITLE_BAR_PADDING)
        title_lines = wrap_text(episode_title, title_font, max_title_width, draw)

        # Calculate positioning
        episode_bbox = draw.textbbox((0, 0), episode_text, font=episode_font)
        episode_height = episode_bbox[3] - episode_bbox[1]
        episode_width = episode_bbox[2] - episode_bbox[0]
        title_line_height = title_font_size + line_spacing
        total_content_height = episode_height + episode_title_gap + (len(title_lines) * title_line_height)
        content_start_y = bar_y + (bar_height - total_content_height) // 2

        # Draw episode number
        if is_square:
            episode_x = (img.size[0] - episode_width) // 2  # Centered
        else:
            episode_x = TITLE_BAR_PADDING  # Left-aligned

        draw.text((episode_x, content_start_y), episode_text, font=episode_font, fill="white")

        # Draw title lines
        title_y = content_start_y + episode_height + episode_title_gap
        for line in title_lines:
            if is_square:
                line_width = draw.textbbox((0, 0), line, font=title_font)[2] - draw.textbbox((0, 0), line, font=title_font)[0]
                line_x = (img.size[0] - line_width) // 2  # Centered
            else:
                line_x = TITLE_BAR_PADDING  # Left-aligned

            draw.text((line_x, title_y), line, font=title_font, fill="white")
            title_y += title_line_height

        # Add logo in bottom-right corner
        logo = Image.open(LOGO_PATH).convert('RGBA')
        logo = logo.resize(logo_size, Image.Resampling.LANCZOS)
        logo_x = img.size[0] - logo_size[0] - TITLE_BAR_PADDING
        logo_y = img.size[1] - logo_size[1] - TITLE_BAR_PADDING
        img.paste(logo, (logo_x, logo_y), logo)

        # Convert to RGB for JPEG
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img

        # Atomic write
        atomic_write(output_path, lambda path: img.save(path, 'JPEG', quality=95, optimize=True))
        print(f"  ✓ Saved: {output_path}")
        return True

    except (OSError, IOError, ValueError) as e:
        logger.error(f"Post-processing failed: {e}")
        return False


# ============================================================================
# ORCHESTRATION AND WORKFLOW
# ============================================================================

async def generate_single_variant(
    session: aiohttp.ClientSession,
    api_key: str,
    provider: Provider,
    concept: str,
    episode_num: int,
    episode_title: str,
    output_dir: Path,
    variant_num: int,
    semaphore: asyncio.Semaphore = None
) -> Tuple[int, Optional[Path], Optional[Path]]:
    """Generate a single variant (both square and social)

    Returns tuple of (variant_num, square_path, social_path)
    """

    async with semaphore if semaphore else asyncio.Lock():
        include_bolt = True  # Always include Bolt
        include_hosts = variant_num >= 3  # Hosts only in variants 3 and 4
        character_label = "bolt+hosts" if include_hosts else "bolt-only"
        title_slug = slugify_title(episode_title)

        print(f"  🎨 {provider.value.capitalize()} variant {variant_num}/4 ({character_label})...")

        if provider == Provider.OPENAI:
            prompt = build_image_prompt(concept, ImageVariant.SQUARE, Provider.OPENAI, include_bolt, include_hosts)
            base_image_bytes = await retry_with_backoff(generate_image_openai, session, api_key, prompt, ImageVariant.SQUARE, concept, include_bolt, include_hosts)
        else:  # GEMINI
            prompt = build_image_prompt(concept, ImageVariant.SQUARE, Provider.GEMINI, include_bolt, include_hosts)
            base_image_bytes = await retry_with_backoff(generate_image_gemini, session, api_key, prompt, ImageVariant.SQUARE, concept, include_bolt, include_hosts)

        if not base_image_bytes:
            print(f"  ⚠️  {provider.value.capitalize()} variant {variant_num} generation failed")
            return (variant_num, None, None)

        # Save square variant
        filename = f"{episode_num}-{title_slug}-{provider.value}-{variant_num}.jpg"
        output_path = output_dir / filename
        square_path = output_path if process_and_save(base_image_bytes, episode_num, episode_title, ImageVariant.SQUARE, output_path) else None

        # Reuse same image for social variant (different cropping/processing)
        filename_social = f"{episode_num}-{title_slug}-social-{provider.value}-{variant_num}.jpg"
        output_path_social = output_dir / filename_social
        social_path = output_path_social if process_and_save(base_image_bytes, episode_num, episode_title, ImageVariant.SOCIAL, output_path_social) else None

        return (variant_num, square_path, social_path)


async def generate_all_variants(
    session: aiohttp.ClientSession,
    api_key: str,
    provider: Provider,
    concept: str,
    episode_num: int,
    episode_title: str,
    output_dir: Path
) -> Dict[ImageVariant, List[Path]]:
    """Generate 4 different images: 2 with Bolt only, 2 with Bolt + hosts

    Returns both square and social variants for each (8 images total per provider)
    """

    results = {ImageVariant.SQUARE: [], ImageVariant.SOCIAL: []}

    # OpenAI: Generate 2 at a time (rate limit friendly)
    # Gemini: Generate all 4 concurrently (faster)
    concurrency = 2 if provider == Provider.OPENAI else 4
    semaphore = asyncio.Semaphore(concurrency)

    # Generate all 4 variants concurrently (with controlled concurrency)
    tasks = [
        generate_single_variant(session, api_key, provider, concept, episode_num, episode_title, output_dir, variant_num, semaphore)
        for variant_num in range(1, 5)
    ]

    variant_results = await asyncio.gather(*tasks)

    # Collect results in order
    for variant_num, square_path, social_path in sorted(variant_results, key=lambda x: x[0]):
        if square_path:
            results[ImageVariant.SQUARE].append(square_path)
        if social_path:
            results[ImageVariant.SOCIAL].append(social_path)

    return results


async def generate_with_providers(
    episode_num: int,
    episode_title: str,
    selected_concept: str,
    providers: List[Provider]
) -> Dict[Provider, Dict[ImageVariant, List[Path]]]:
    """Generate 4 images per provider: 2 with Bolt only, 2 with Bolt + hosts

    Total output: 8 square + 8 social images per provider
    """

    print(f"\n🖼️  Generating images...")
    print("=" * 70)
    print("Each provider will generate:")
    print("  • 2 images with Bolt only")
    print("  • 2 images with Bolt + all four hosts")
    print("  • Each in both square (3000×3000) and social (1200×630) formats")

    all_results = {}

    # Use episode-numbered subdirectory
    output_dir = OUTPUT_DIR / str(episode_num)
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
    # Convert concepts to dict format for JSON
    concepts_list = [{"concept": concept, "provider": provider} for concept, provider in concepts]
    selected_concept, selected_provider = concepts[selected_index - 1]

    concepts_data = {
        "episode": episode_num,
        "title": episode_title,
        "concepts": concepts_list,
        "selected_index": selected_index,
        "selected_concept": selected_concept,
        "selected_provider": selected_provider
    }

    # Save to episode-numbered subdirectory
    episode_dir = OUTPUT_DIR / str(episode_num)
    concepts_file = episode_dir / f"{episode_num}-concepts.json"

    try:
        atomic_write(concepts_file, lambda path: json.dump(concepts_data, open(path, 'w'), indent=2))
        print(f"  ✓ Concepts saved to: {concepts_file}")
    except Exception as e:
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
                    # Variants 1-2 are Bolt only, 3-4 are Bolt + hosts
                    character_info = "Bolt only" if i <= 2 else "Bolt + hosts"
                    print(f"  ✓ Variant {i} ({character_info}): {path.name} ({size})")
            else:
                print(f"  ✗ {variant.value} generation failed")

    print("\n" + "=" * 70)
    print("\nVariant guide:")
    print("  • Variants 1-2: Bolt only (2 different interpretations)")
    print("  • Variants 3-4: Bolt + all four hosts (2 different interpretations)")
    print("=" * 70)


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
    # Each provider generates 4 variants: 2 with Bolt only, 2 with Bolt + hosts
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
