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
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OpenAI Models - Latest
OPENAI_CHAT_MODEL = "gpt-5.2"  # Latest reasoning model
OPENAI_IMAGE_MODEL = "gpt-image-1.5"  # GPT Image model with up to 16-reference support

# Gemini Models - Gemini 3
GEMINI_TEXT_MODEL = "gemini-3-flash-preview"  # Latest flash with thinking_level
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"  # Nano Banana Pro - 4K with up to 14 references
# NOTE: Using Pro over Flash (Nano Banana 2) - Pro provides superior composition, character design,
# and thinking capability for complex multi-character scenes. Update to next "Pro" model when available.

# Anthropic Claude - Concept Generation Only (no image generation)
ANTHROPIC_CHAT_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_CHAT_MODEL = "claude-sonnet-4-6"  # Fast, creative concept generation

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
SQUARE_BAR_HEIGHT_COMPACT = 350   # Smaller overlay for comic strip layouts (~12% of 3000px)
SOCIAL_BAR_HEIGHT_COMPACT = 95    # Smaller overlay for comic strip layouts (~15% of 630px)
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
ANTHROPIC_TIMEOUT = 120

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

# JPEG Compression (web-optimized)
# Square (3000x3000): quality=85, ~400-700KB (down from 1.3-2.2MB at quality=95)
# Social (1200x630): quality=80, ~100-200KB
# Both use progressive encoding for better web loading

# Retry Configuration
MAX_API_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds

# Concurrency
PROCESS_POOL = ThreadPoolExecutor(max_workers=4)


class AsyncRateLimiter:
    """Ensures a minimum interval between calls (async-safe)"""

    def __init__(self, min_interval_seconds: float):
        self.min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            sleep_for = self.min_interval - (now - self._last)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._last = time.monotonic()


# Rate limiters per provider
openai_image_limiter = AsyncRateLimiter(OPENAI_DELAY)


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
        except Exception as e:
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
    """Load all host images with metadata for multi-reference identity lock

    Returns list of dicts with 'name', 'data' (base64), 'role', 'priority'
    Ordered by priority for reference hierarchy (GPT Image supports up to 16 references)
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


def load_host_references_as_pil() -> List[Dict[str, any]]:
    """Load all host images as PIL Image objects for google-genai SDK

    Returns list of dicts with 'name', 'image' (PIL Image), 'role', 'priority'
    Ordered by priority for reference hierarchy (Gemini 3 SDK)
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

            # Load as PIL Image
            img = Image.open(ref_path)
            loaded_refs.append({
                "name": ref_spec["name"],
                "image": img,
                "role": ref_spec["role"],
                "priority": ref_spec["priority"],
                "filename": ref_spec["filename"]
            })

        if loaded_refs:
            print(f"  ✓ Loaded {len(loaded_refs)} reference images as PIL objects in priority order:")
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


# Creative lenses for concept variety - each produces a fundamentally different type of concept
CREATIVE_LENSES = [
    {
        "name": "pop_culture_parody",
        "label": "Pop Culture Parody",
        "instruction": """CREATIVE LENS: POP CULTURE PARODY
First, ask yourself: Does this title reference a movie, TV show, meme, famous quote, or cultural moment?
If so, create a concept that PARODIES that reference. Reimagine the iconic scene/moment with Cloud Pod characters and tech humor.
If there's no obvious reference, parody a well-known visual format (movie poster, album cover, famous painting, meme template) that fits the title's mood.
Examples:
- "AI'll be Back" → Bolt as the Terminator in sunglasses, walking away from an explosion, chrome hand giving thumbs up
- "The Great Migration" → Hosts as wildebeest crossing a river of cloud logos, nature-documentary style
- "Jimmy Droptables" → Courtroom drama parody - Jimmy on trial, evidence table literally collapsing"""
    },
    {
        "name": "hero_shot",
        "label": "Hero Shot",
        "instruction": """CREATIVE LENS: HERO SHOT
Create a SIMPLE, bold composition with ONE main subject, ONE clear action, and a strong background.
Think podcast cover or movie poster - this should read instantly at thumbnail size.
Keep it to 3 elements max. No busy scenes, no Rube Goldberg chains of events.
The image should tell the story through character expression, pose, and a single strong visual metaphor.
Examples:
- "Just-in-Time Secrets" → Bolt mid-sneak through a vault door, clutching a folder, guilty grin
- "Scaling Down" → Single towering building shrinking like a deflating balloon, Bolt watching from below
- "AI'll be Back" → Bolt in aviator sunglasses, half his face chrome circuitry, staring at camera"""
    },
    {
        "name": "genre_shift",
        "label": "Genre Shift",
        "instruction": """CREATIVE LENS: GENRE SHIFT
Take the title's story and tell it in a COMPLETELY UNEXPECTED genre or setting. Drop the tech/office context entirely.
Reimagine it as: a Western, nature documentary, cooking show, sports broadcast, fairy tale, noir detective story, space opera, medieval quest, courtroom drama, heist film, or any other genre that creates a funny contrast.
The humor comes from the clash between the tech topic and the unexpected world.
Examples:
- "Container Orchestration" → Medieval conductor Bolt directing an orchestra of shipping containers in a concert hall
- "AWS Layoffs" → Old Western ghost town, tumbleweeds rolling past abandoned "AWS Saloon"
- "Zero Trust Security" → Film noir detective scene, everyone in trench coats eyeing each other suspiciously"""
    },
    {
        "name": "storytelling_moment",
        "label": "Storytelling Moment",
        "instruction": """CREATIVE LENS: STORYTELLING MOMENT
Show a HUMAN MOMENT that tells the real-world story behind the title. No visual puns, no mechanisms, no clever wordplay.
Think photojournalism or editorial illustration: what would an artist draw to capture the REAL story this episode is about?
Characters should be REACTING TO or WITNESSING events, not causing them. Bolt and the hosts can be observers, bystanders, or participants in an emotional scene.
Zero text labels. The scene speaks for itself through body language, setting, and composition.
Examples:
- "AWS Layoffs: Scaling Down" → Somber employees with cardboard boxes filing out of a glass office tower, the Cloud Pod hosts and Bolt watching from the sidewalk
- "Just-in-Time Secrets" → Bolt sitting alone at a desk at 2am, screen glow on face, surrounded by scattered confidential folders
- "The Great Cloud Migration" → Caravan of people carrying servers on their backs, walking a long road toward a distant glowing cloud city"""
    },
    {
        "name": "visual_pun",
        "label": "Visual Pun",
        "instruction": """CREATIVE LENS: VISUAL PUN
Find the wordplay or double meaning in the title and make it LITERALLY VISUAL.
Take one or two key words and show their alternate meaning as a physical reality.
Keep it focused on ONE strong pun rather than trying to literalize every word.
Limit to 1-2 small text labels max - if the image needs labels to be funny, the visual isn't strong enough.
Examples:
- "Jimmy Droptables" → Bolt yanking a tablecloth off a fancy dinner table, database records flying like dishes
- "Scaling Down" → Bolt on a fish being de-scaled with a giant scraper, tiny employees falling off like scales
- "Container Orchestration" → Bolt conducting an orchestra where all the instruments are shipping containers"""
    },
    {
        "name": "minimalist",
        "label": "Minimalist / Abstract",
        "instruction": """CREATIVE LENS: MINIMALIST / ABSTRACT
Distill the title down to ONE iconic image. Think New Yorker cover, editorial illustration, or graphic design poster.
Use negative space, bold color, and a single striking visual element. No complex scenes or multiple characters.
This should be visually elegant and immediately readable. Fewer elements = stronger impact.
Examples:
- "AI'll be Back" → Single chrome endoskeleton hand emerging from a cloud, giving thumbs up
- "Zero Trust" → A single handshake where both hands are crossed fingers behind their backs
- "Serverless Everything" → Pristine empty server rack with a single tumbleweed rolling through"""
    },
    {
        "name": "mood_piece",
        "label": "Mood & Atmosphere",
        "instruction": """CREATIVE LENS: MOOD & ATMOSPHERE
Instead of explaining the joke, FEEL it. Create a scene driven by emotion, lighting, and atmosphere rather than clever mechanisms.
What's the emotional core of this title? Capture that feeling through dramatic lighting, color temperature, and character body language.
This can be dramatic, ironic, melancholy, suspenseful, triumphant, eerie, or absurdly serene.
No text labels in the scene. No visual puns. No mechanisms or levers. Just a moment and a mood.
Examples:
- "Just-in-Time Secrets" → Dark datacenter corridor, single spotlight on Bolt tiptoeing past, long dramatic shadows
- "AI'll be Back" → Foggy server room at night, single red eye glowing in the distance, ominous silhouette
- "Container Orchestration" → Vast harbor at golden hour, thousands of shipping containers stretching to the horizon, Bolt conducting from a tiny podium"""
    },
    {
        "name": "everyday_metaphor",
        "label": "Everyday Metaphor",
        "instruction": """CREATIVE LENS: EVERYDAY METAPHOR
Find a completely UNRELATED real-world situation that captures the same FEELING or DYNAMIC as the title.
Don't literalize the tech words. Instead, ask: "What everyday experience feels like this?"
Map the tech concept onto something warm, human, and immediately relatable - no tech knowledge needed to get the joke.
The humor comes from the unexpected parallel between the tech story and the mundane situation.
Examples:
- "AWS Layoffs: Scaling Down" → A gardener tenderly pruning back a massive overgrown hedge, clippings piling up
- "Prices Go Both Ways, Raises GPU Costs" → A kid's helium balloon slipping from their hand and floating away while they reach for it
- "Crawlers Running the Asylum" → A golden retriever sitting in the driver's seat of a car, paws on the wheel, looking way too confident
- "Oracle Discovers the Dark Side" → A kid in pajamas sneaking cookies from a jar after bedtime"""
    },
    {
        "name": "comic_strip",
        "label": "Comic Strip",
        "instruction": """CREATIVE LENS: COMIC STRIP
Create a 4-PANEL comic strip that tells the episode's story as a sequential narrative with a punchline.
Layout: 2x2 grid (two panels on top, two on bottom). The BOTTOM of the image will be covered by the show title overlay, so:
- TOP TWO PANELS: Set up the joke (panels 1 and 2, these are fully visible)
- BOTTOM TWO PANELS: Deliver the punchline in the UPPER PORTION of these panels. Keep the very bottom edge of the bottom panels unimportant (simple backgrounds, no faces or key details down there)
Each panel should be DEAD SIMPLE - one character, one action, minimal background. Think XKCD, Dilbert, or The Oatmeal.
Short speech bubbles or captions are fine (this is a comic after all), but keep text minimal - 3-5 words per panel max.
Describe all 4 panels in order: Panel 1 (top-left), Panel 2 (top-right), Panel 3 (bottom-left), Panel 4 (bottom-right).
Examples:
- "AWS Layoffs: Scaling Down" → Panel 1: Bolt at whiteboard writing "Scale OUT!" / Panel 2: Boss taps shoulder, points at budget chart / Panel 3: Bolt erases "OUT", writes "Down" / Panel 4: Bolt alone in a huge empty office, tiny wave
- "Jimmy Droptables" → Panel 1: "Meet our new SQL assistant!" / Panel 2: Jimmy smiles, types "DROP TABLE" / Panel 3: Hosts stare in horror at empty screen / Panel 4: Jimmy shrugs, "Feature, not a bug" """
    },
]


def build_concept_prompt(episode_title: str, previous_concepts: List[str] = None, keywords: str = None, creative_lens: dict = None) -> str:
    """Build prompt for concept generation (text-only phase)

    Args:
        episode_title: The episode title to generate concepts for
        previous_concepts: List of previously generated concept texts to avoid duplicating
        keywords: Optional keywords to steer creative direction
        creative_lens: Optional dict with 'name', 'label', and 'instruction' keys
                      specifying the creative approach for this concept
    """

    # Build dynamic sections
    sections = ["""You are the creative director for The Cloud Pod, a tech podcast. You create bold, varied visual concepts for podcast cover art."""]

    # Detect if title mentions AI/agent/bot/robot - these should map to Bolt
    ai_keywords = ['ai', 'agent', 'bot', 'robot', 'artificial', 'intelligence', 'llm', 'model', 'chatbot']
    title_lower = episode_title.lower()
    mentions_ai = any(keyword in title_lower for keyword in ai_keywords)

    sections.append(f"""

Episode Title: '{episode_title}'

YOUR TASK: Create ONE completely NEW visual concept for a podcast cover based on '{episode_title}'.

BEFORE YOU START: Ask yourself these questions about the title:
1. Is it referencing a movie, meme, saying, or cultural moment? If so, consider riffing on that.
2. Is there a double meaning or wordplay? What's the joke underneath the surface?
3. What FEELING does this title evoke? (funny, ominous, triumphant, absurd, bittersweet?)
4. What's the simplest possible image that captures the essence?""")

    # Add the specific creative lens if provided
    if creative_lens:
        sections.append(f"""

{creative_lens['instruction']}""")
    else:
        # Fallback if no lens specified (e.g., for "generate more" or custom flows)
        sections.append("""

APPROACH: Choose the single best creative angle for this title. You might:
- Parody a pop culture reference the title is making
- Create a simple hero shot with one character and one strong visual
- Shift to an unexpected genre (Western, noir, fairy tale, nature documentary)
- Capture the mood/emotion rather than explaining the joke
- Find a visual pun on one key word
- Go minimalist with one bold iconic image""")

    sections.append(f"""

QUALITY GUIDELINES:
- Be SPECIFIC about the scene (what do we see? where is it? what's the lighting?) but keep it SIMPLE - 3-5 key elements max
- The concept should read at thumbnail size. If the scene is too busy to grasp in one glance, simplify
- Maximum 1-2 text labels in the scene. If the image needs labels to make sense, the visual isn't strong enough. Many great concepts have ZERO labels
- Characters don't always need to be the ACTIVE AGENT doing something clever. Sometimes the most powerful image is characters WITNESSING, REACTING TO, or simply EXISTING IN a scene
- Do NOT reuse visual elements from other episodes

IMPORTANT:
- Do NOT just decompose the title word-by-word into literal objects arranged in a scene
- Do NOT default to "Bolt pulls/pushes/operates a [mechanism]" - that's a crutch
- Find the deeper joke, the cultural reference, or the emotional core of the title""")

    if previous_concepts:
        concepts_list = "\n".join(f"{i}. {c}" for i, c in enumerate(previous_concepts, 1))
        sections.append(f"""

PREVIOUS CONCEPTS ALREADY GENERATED (do NOT repeat these ideas or use the same creative approach):
{concepts_list}

Your concept must use a FUNDAMENTALLY DIFFERENT creative angle from these - not just different props in the same type of scene.""")

    if keywords and keywords.strip():
        sections.append(f"""

KEYWORD GUIDANCE: The user wants concepts that incorporate or emphasize these themes: {keywords.strip()}
Use these keywords to steer your creative direction.""")

    # Dynamic character guidance based on title content
    if mentions_ai:
        sections.append(f"""

CHARACTER GUIDANCE:
This title mentions AI/agents/bots. Our mascot **Bolt** (blue cloud robot) naturally represents AI in our universe.
- When the title says "AI Agent" or similar, Bolt IS that agent
- **The Four Hosts** (Jonathan, Justin, Matthew, Ryan) are available for human character dynamics
- You can also create concepts with NO characters - object-only or environmental scenes work too""")
    else:
        sections.append(f"""

AVAILABLE CHARACTERS (use when they enhance the concept, or skip them entirely):
- **Bolt** - Blue cloud robot mascot - great for comedy and character-driven concepts
- **The Four Hosts** - Jonathan, Justin, Matthew, Ryan - for human-focused scenes
- No characters at all - object-only, environmental, or abstract concepts are equally valid""")

    # Tailor the output format instruction to the lens type
    if creative_lens and creative_lens['name'] == 'comic_strip':
        sections.append(f"""

Describe all 4 panels concisely. Use the format: "Panel 1: ... / Panel 2: ... / Panel 3: ... / Panel 4: ..."
Keep each panel description to one short sentence. No preamble, no explanation - just the panel descriptions.""")
    else:
        sections.append(f"""

Return ONLY the concept as a single concise sentence describing the visual scene. No preamble, no explanation, no alternatives - just the concept itself.""")

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

    # Detect comic strip concepts and adjust instructions
    is_comic_strip = concept.strip().lower().startswith("panel 1")

    # For comic strips, override the composition guidance to work with 2x2 grid
    if is_comic_strip:
        dimension_guidance = f"""{format_type}

COMIC STRIP LAYOUT:
This concept is a 4-panel comic strip. Render it as a 2x2 grid that fits ENTIRELY in the UPPER 88% of the image.

PANEL ARRANGEMENT:
- TOP ROW: Panel 1 (top-left) and Panel 2 (top-right) - setup panels
- BOTTOM ROW: Panel 3 (bottom-left) and Panel 4 (bottom-right) - punchline panels
- Clear visible borders/gutters between all 4 panels

PANEL PROPORTIONS: Each panel is a WIDE HORIZONTAL RECTANGLE (roughly 2:1 width-to-height), like widescreen movie frames or newspaper comic strip panels. NOT square panels.

DEAD ZONE: The bottom ~12% of the image will be covered by a compact title overlay.
- Leave the bottom ~12% as simple solid background color, panel border color, or empty space
- ALL four panels including panels 3 and 4 must be FULLY VISIBLE above this dead zone
- No faces, speech bubbles, or important details in the bottom 12%

TEXT IN IMAGE: ONLY render text that appears in the panel descriptions (speech bubbles, captions, on-screen labels). Do NOT render any character descriptions, instructions, or prompt text as visible text in the image."""
    else:
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

    # For comic strips, add a reminder not to render character descriptions as image text
    comic_text_warning = ""
    if is_comic_strip:
        comic_text_warning = """

TEXT RENDERING WARNING: The character descriptions below are RENDERING INSTRUCTIONS for how to draw the characters. Do NOT display any of this descriptive text in the image. Only render text that appears in speech bubbles or captions as described in the panel concept."""

    return f"""{BASE_STYLE_PROMPT}

SPECIFIC EPISODE CONCEPT:
{concept}

CRITICAL: Render the elements described in the concept above. Do not add unrelated objects or text that are not explicitly mentioned in the concept.{character_requirement}{comic_text_warning}
{bolt_guidance}
{hosts_guidance}

COMPOSITION REQUIREMENTS:
{dimension_guidance}

Generate a background image that visualizes this concept while maintaining The Cloud Pod's professional tech aesthetic.{model_emphasis}"""


# ============================================================================
# TEXT GENERATION FUNCTIONS (Concept Phase)
# ============================================================================

def clean_concept_text(text: str) -> Optional[str]:
    """Clean and validate concept text, stripping thinking artifacts and truncation.

    Returns cleaned text or None if the concept is malformed beyond repair.
    """
    if not text or not text.strip():
        return None

    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove markdown formatting artifacts (bold, italic markers)
    text = text.strip('*').strip('_').strip()

    # Detect Gemini thinking leakage: deliberation text with bullet points, questions, alternatives
    thinking_indicators = [
        '* ', '*\t', '    *', 'How about', 'What about', 'What if',
        'Let me think', 'I could', 'Option ', 'Alternatively',
    ]
    lines = text.split('\n')
    # If more than half the lines look like thinking, reject the whole concept
    thinking_lines = sum(1 for line in lines if any(line.strip().startswith(ind) for ind in thinking_indicators))
    if len(lines) > 1 and thinking_lines > len(lines) // 2:
        logger.warning("Rejected concept: appears to contain model thinking/deliberation")
        return None

    # If the text starts mid-thought (e.g., "different*.\n    *   How about"), reject it
    if text.startswith(('different', 'instead', 'rather', 'but ', 'however', 'or ')):
        logger.warning("Rejected concept: starts mid-thought")
        return None

    # Extract just the first coherent concept if thinking is appended after it
    # Look for a clean sentence before any deliberation starts
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and any(stripped.startswith(ind) for ind in thinking_indicators):
            # Everything before this line is the real concept
            clean_part = '\n'.join(lines[:i]).strip()
            if clean_part and len(clean_part) > 30:
                text = clean_part
                break

    # Check for truncation: concept should end with proper punctuation
    if text and text[-1] not in '.!?""\u201d)':
        # Concept appears truncated - try to salvage by trimming to last complete sentence
        last_period = max(text.rfind('. '), text.rfind('.\n'), text.rfind('."'), text.rfind('.\u201d'))
        if last_period > len(text) * 0.5:  # Only trim if we keep at least half
            text = text[:last_period + 1]
            logger.warning("Trimmed truncated concept to last complete sentence")
        else:
            logger.warning("Rejected concept: appears truncated with no recovery point")
            return None

    # Final length check - a concept should be at least a reasonable sentence
    if len(text) < 30:
        logger.warning(f"Rejected concept: too short ({len(text)} chars)")
        return None

    return text


async def _make_api_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
    provider_name: str,
    extract_response,
    timeout: int = None
) -> Optional[str]:
    """Generic API request handler for OpenAI, Gemini, and Anthropic"""
    # Use provider-specific timeout, falling back to provider defaults
    if timeout is None:
        if provider_name == "OpenAI":
            timeout = OPENAI_TIMEOUT
        elif provider_name == "Gemini":
            timeout = GEMINI_TIMEOUT
        else:
            timeout = ANTHROPIC_TIMEOUT
    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status == 200:
                data = await response.json()
                raw_text = extract_response(data)
                # Validate and clean the concept text
                cleaned = clean_concept_text(raw_text)
                if cleaned is None:
                    logger.warning(f"{provider_name} returned malformed concept, retrying...")
                    return None
                return cleaned
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
    keywords: str = None,
    creative_lens: dict = None
) -> Optional[str]:
    """Generate a creative concept using OpenAI chat completions"""
    payload = {
        "model": OPENAI_CHAT_MODEL,
        "messages": [{"role": "user", "content": build_concept_prompt(episode_title, previous_concepts, keywords, creative_lens)}],
        "max_completion_tokens": 2000,
        "reasoning_effort": "medium"
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=OPENAI_TIMEOUT)
        ) as response:
            if response.status == 200:
                data = await response.json()
                raw_text = data["choices"][0]["message"]["content"].strip()
                cleaned = clean_concept_text(raw_text)
                if cleaned is None:
                    logger.warning("OpenAI returned malformed concept, retrying...")
                    return None
                return cleaned
            else:
                error_text = await response.text()
                logger.error(f"OpenAI concept request failed: {error_text}")
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"OpenAI concept request error: {e}")
        return None


async def generate_concept_gemini(
    session: aiohttp.ClientSession,
    api_key: str,
    episode_title: str,
    previous_concepts: List[str] = None,
    keywords: str = None,
    creative_lens: dict = None
) -> Optional[str]:
    """Generate a creative concept using Google Gemini SDK with Flash 3"""

    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    prompt = build_concept_prompt(episode_title, previous_concepts, keywords, creative_lens)

    # Configure generation for creative text generation
    config = types.GenerateContentConfig(
        temperature=0.9,
        max_output_tokens=2048
    )

    try:
        # Use Gemini 3 Flash for fast, creative concept generation
        response = client.models.generate_content(
            model=GEMINI_TEXT_MODEL,  # "gemini-3-flash-preview"
            contents=prompt,
            config=config
        )

        # Extract text from response
        if response.candidates and len(response.candidates) > 0:
            text = response.candidates[0].content.parts[0].text.strip()

            # Validate and clean the concept text
            cleaned = clean_concept_text(text)
            if cleaned is None:
                logger.warning("Gemini returned malformed concept, retrying...")
                return None
            return cleaned

        logger.error("No text in Gemini SDK response")
        return None

    except Exception as e:
        logger.error(f"Gemini SDK request error: {e}")
        return None


async def generate_concept_anthropic(
    session: aiohttp.ClientSession,
    api_key: str,
    episode_title: str,
    previous_concepts: List[str] = None,
    keywords: str = None,
    creative_lens: dict = None
) -> Optional[str]:
    """Generate a creative concept using Anthropic Claude"""
    payload = {
        "model": ANTHROPIC_CHAT_MODEL,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": build_concept_prompt(episode_title, previous_concepts, keywords, creative_lens)}]
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    return await _make_api_request(
        session, ANTHROPIC_CHAT_ENDPOINT, payload, headers, "Anthropic",
        lambda data: data["content"][0]["text"].strip()
    )


async def generate_text_with_provider(prompt: str, provider: str, max_tokens: int = 512, temperature: float = 0.7) -> Optional[str]:
    """Helper function to generate text with any provider (uses SDK for Gemini)

    Args:
        prompt: The text prompt to send
        provider: "OpenAI", "Anthropic", or "Gemini"
        max_tokens: Maximum output tokens
        temperature: Sampling temperature

    Returns:
        Generated text or None on error
    """
    try:
        if provider == "OpenAI" and OPENAI_API_KEY:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": OPENAI_CHAT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": max_tokens
                    },
                    timeout=aiohttp.ClientTimeout(total=OPENAI_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"].strip()

        elif provider == "Anthropic" and ANTHROPIC_API_KEY:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ANTHROPIC_CHAT_ENDPOINT,
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": ANTHROPIC_CHAT_MODEL,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=aiohttp.ClientTimeout(total=ANTHROPIC_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["content"][0]["text"].strip()

        elif provider == "Gemini" and GOOGLE_API_KEY:
            # Use the official SDK for Gemini
            client = genai.Client(api_key=GOOGLE_API_KEY)
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )

            response = client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=prompt,
                config=config
            )

            if response.candidates and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text.strip()

        logger.error(f"Text generation with {provider} failed or no API key")
        return None

    except Exception as e:
        logger.error(f"Text generation error with {provider}: {e}")
        return None


async def generate_concepts(episode_title: str) -> List[Tuple[str, str]]:
    """Generate one concept per creative lens, split across providers.
    Each concept uses a different creative lens to guarantee structural diversity.
    Returns list of tuples: (concept_text, provider_name)
    """

    num_lenses = len(CREATIVE_LENSES)
    print(f"\n🎨 Generating {num_lenses} creative concepts for: \"{episode_title}\"")
    print("=" * 70)

    concepts = []
    previous_concepts = []

    async with aiohttp.ClientSession() as session:
        # Build provider list - one concept per lens, rotating across available providers
        available_providers = []
        if OPENAI_API_KEY:
            available_providers.append(("OpenAI", generate_concept_openai, OPENAI_API_KEY))
        if GOOGLE_API_KEY:
            available_providers.append(("Gemini", generate_concept_gemini, GOOGLE_API_KEY))
        if ANTHROPIC_API_KEY:
            available_providers.append(("Anthropic", generate_concept_anthropic, ANTHROPIC_API_KEY))

        providers = []
        for i, lens in enumerate(CREATIVE_LENSES):
            prov_name, prov_func, prov_key = available_providers[i % len(available_providers)]
            providers.append((prov_name, prov_func, prov_key, lens))

        for provider_name, generate_func, api_key, lens in providers:
            print(f"  Generating {provider_name} concept {len(concepts) + 1}/{num_lenses} [{lens['label']}]...")
            result = await retry_with_backoff(
                generate_func, session, api_key, episode_title, previous_concepts,
                None,  # keywords
                lens   # creative_lens
            )

            if isinstance(result, str) and result:
                concepts.append((result, provider_name))
                previous_concepts.append(result)
            else:
                print(f"  ⚠️  {provider_name} concept generation failed")

    # Require at least half the lenses to succeed
    min_concepts = num_lenses // 2
    if len(concepts) < min_concepts:
        print(f"\n❌ Error: Only generated {len(concepts)} concepts, need at least {min_concepts}")
        print("Please check API keys and try again.")
        return []

    return concepts


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

    # Pick creative lenses that haven't been used yet, cycling through if needed
    num_existing = len(existing_concepts)
    lenses_for_more = []
    for i in range(count):
        lens_idx = (num_existing + i) % len(CREATIVE_LENSES)
        lenses_for_more.append(CREATIVE_LENSES[lens_idx])

    async with aiohttp.ClientSession() as session:
        # Rotate across available providers for variety
        available_providers = []
        if OPENAI_API_KEY:
            available_providers.append(("OpenAI", generate_concept_openai, OPENAI_API_KEY))
        if GOOGLE_API_KEY:
            available_providers.append(("Gemini", generate_concept_gemini, GOOGLE_API_KEY))
        if ANTHROPIC_API_KEY:
            available_providers.append(("Anthropic", generate_concept_anthropic, ANTHROPIC_API_KEY))

        providers = []
        for i in range(count):
            lens = lenses_for_more[i]
            prov_name, prov_func, prov_key = available_providers[i % len(available_providers)]
            providers.append((prov_name, prov_func, prov_key, lens))

        for provider_name, generate_func, api_key, lens in providers:
            print(f"  Generating {provider_name} concept {len(new_concepts) + 1}/{count} [{lens['label']}]...")
            result = await retry_with_backoff(
                generate_func,
                session,
                api_key,
                episode_title,
                previous_concepts,
                keywords,  # Pass keywords for steering
                lens       # creative_lens
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

    return await generate_text_with_provider(prompt, provider, max_tokens=CONCEPT_REFINEMENT_TOKENS, temperature=0.7)


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

IMPORTANT PRESERVATION RULES:
- Keep the user's core idea intact. Only enhance and clarify, don't change the fundamental concept.
- If the user uses "Panel 1:", "Panel 2:", "Panel 3:", "Panel 4:" format for a comic strip, PRESERVE that exact format. Do NOT change it to (1), (2), (3), (4) or any other numbering style.

Return ONLY the polished concept:"""

    # Use first available provider: OpenAI > Anthropic > Gemini
    if OPENAI_API_KEY:
        provider = "OpenAI"
    elif ANTHROPIC_API_KEY:
        provider = "Anthropic"
    else:
        provider = "Gemini"

    return await generate_text_with_provider(prompt, provider, max_tokens=CONCEPT_REFINEMENT_TOKENS, temperature=0.7)


async def present_concepts_and_choose(concepts: List[Tuple[str, str]], episode_title: str) -> Tuple[List[Tuple[int, str]], bool]:
    """Display concepts and get user selection (supports multi-select)

    Args:
        concepts: List of (concept_text, provider_name) tuples (can grow beyond initial set)
        episode_title: Episode title for refinement prompts

    Returns:
        Tuple of (selections, should_regenerate) where selections is a list of (index, concept_text) tuples
    """

    while True:  # Outer loop to handle concept additions
        print("\n📋 Creative Concepts:")
        print("=" * 70)

        for i, (concept, provider) in enumerate(concepts, 1):
            print(f"\n{i}. [{provider}] {concept}")

        print("\n" + "=" * 70)
        concept_range = f"1-{len(concepts)}"
        refine_range = f"R1-R{len(concepts)}"
        print(f"Commands: [{concept_range}] = Select concept(s) (e.g. 3 or 1,3,7) | W = Write your own | 0 = Regenerate | M = More concepts | {refine_range} = Refine | X = Exit")

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
                    return [], True

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
                                    # Ask if it's a comic strip
                                    is_comic = input("\nIs this a comic strip / multi-panel layout? (y/n): ").strip().lower()
                                    if is_comic == 'y' and "panel" not in edited_concept.lower():
                                        edited_concept = f"Panel 1: {edited_concept}"
                                    concepts.append((edited_concept, "Custom"))
                                    return [(len(concepts), edited_concept)], False
                                else:
                                    is_comic = input("\nIs this a comic strip / multi-panel layout? (y/n): ").strip().lower()
                                    if is_comic == 'y' and "panel" not in polished.lower():
                                        polished = f"Panel 1: {polished}"
                                    concepts.append((polished, "Custom"))
                                    return [(len(concepts), polished)], False
                            elif confirm != 'n':
                                is_comic = input("\nIs this a comic strip / multi-panel layout? (y/n): ").strip().lower()
                                if is_comic == 'y' and "panel" not in polished.lower():
                                    polished = f"Panel 1: {polished}"
                                concepts.append((polished, "Custom"))
                                return [(len(concepts), polished)], False
                            else:
                                use_original = input("\nUse your original concept instead? (Y/n): ").strip().lower()
                                if use_original != 'n':
                                    is_comic = input("\nIs this a comic strip / multi-panel layout? (y/n): ").strip().lower()
                                    if is_comic == 'y' and "panel" not in custom_concept.lower():
                                        custom_concept = f"Panel 1: {custom_concept}"
                                    concepts.append((custom_concept, "Custom"))
                                    return [(len(concepts), custom_concept)], False
                        else:
                            print("\n⚠️  Polishing failed, but you can still use your original concept.")
                            confirm = input("Use your original concept? (Y/n): ").strip().lower()
                            if confirm != 'n':
                                is_comic = input("\nIs this a comic strip / multi-panel layout? (y/n): ").strip().lower()
                                if is_comic == 'y' and "panel" not in custom_concept.lower():
                                    custom_concept = f"Panel 1: {custom_concept}"
                                concepts.append((custom_concept, "Custom"))
                                return [(len(concepts), custom_concept)], False
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
                                        return [(refine_num, refined)], False
                            continue
                        else:
                            print(f"Please enter R1-R{len(concepts)} to refine a concept")
                            continue
                    except ValueError:
                        print(f"Please enter R# where # is 1-{len(concepts)} (e.g., R3)")
                        continue

                # Select concept(s) - supports "3" or "1,3,7"
                # Parse comma-separated numbers
                selections = []
                parts = choice.replace(' ', '').split(',')
                valid = True
                for part in parts:
                    try:
                        num = int(part)
                        if 1 <= num <= len(concepts):
                            concept_text, provider = concepts[num - 1]
                            selections.append((num, concept_text))
                        else:
                            print(f"❌ Concept {num} is out of range (1-{len(concepts)})")
                            valid = False
                            break
                    except ValueError:
                        valid = False
                        break

                if valid and selections:
                    if len(selections) == 1:
                        idx, text = selections[0]
                        provider = concepts[idx - 1][1]
                        print(f"\n✓ Selected #{idx} [{provider}]: {text}\n")
                    else:
                        print(f"\n✓ Selected {len(selections)} concepts:")
                        for idx, text in selections:
                            provider = concepts[idx - 1][1]
                            print(f"  #{idx} [{provider}]: {text[:80]}...")
                        print()
                    return selections, False
                elif not valid:
                    print(f"Please enter 1-{len(concepts)} (or comma-separated like 1,3,7), W, 0, M, R#, or X")

            except ValueError:
                print(f"Please enter 1-{len(concepts)} (or comma-separated like 1,3,7), W, 0, M, R#, or X")
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


async def generate_image_openai(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    variant: ImageVariant,
    concept: str = "",
    include_bolt: bool = False,
    include_hosts: bool = False
) -> Optional[bytes]:
    """Generate image using OpenAI GPT Image 1.5 via official SDK

    Uses images.edit with up to 16 reference images for identity lock,
    or images.generate when no references are needed.
    """

    print(f"  🎨 OpenAI GPT-Image-1.5 generating {variant.value} variant...")

    # Rate limit to stay within OpenAI's per-minute quota
    await openai_image_limiter.wait()

    client = AsyncOpenAI(api_key=api_key)
    use_edit_endpoint = include_bolt or include_hosts

    try:
        if use_edit_endpoint:
            # Load and filter references
            references = load_host_references()
            filtered_refs = _filter_references(references, include_bolt, include_hosts)

            print(f"  📋 Loaded {len(references)} total references, filtered to {len(filtered_refs)} (Bolt={include_bolt}, Hosts={include_hosts})")

            # Build identity lock instructions and add to prompt
            reference_instructions = build_reference_hierarchy_instructions(references, include_bolt, include_hosts)
            enhanced_prompt = f"""{reference_instructions}

GENERATION TASK:
{prompt}"""

            # Open reference images as file objects for the SDK
            hosts_dir = SCRIPT_DIR / "Hosts"
            image_files = []
            for ref in filtered_refs:
                ref_path = hosts_dir / ref["filename"]
                image_files.append(open(ref_path, "rb"))

            print(f"  📸 Using {len(image_files)} reference images with identity lock")

            try:
                result = await client.images.edit(
                    model=OPENAI_IMAGE_MODEL,
                    image=image_files,  # supports up to 16 images
                    prompt=enhanced_prompt,
                    size="1024x1024",
                    quality="high",
                    input_fidelity="high",
                )
            finally:
                for f in image_files:
                    f.close()

            img_bytes = base64.b64decode(result.data[0].b64_json)
            print(f"  ✓ OpenAI {variant.value} generated")
            return img_bytes

        else:
            # Use generate endpoint (no reference images)
            if include_hosts:
                print("  📝 Using text descriptions for 4 hosts (Jonathan, Justin, Matthew, Ryan)")

            result = await client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size="1024x1024",
                quality="high",
            )

            img_bytes = base64.b64decode(result.data[0].b64_json)
            print(f"  ✓ OpenAI {variant.value} generated")
            return img_bytes

    except Exception as e:
        logger.error(f"OpenAI image generation error: {e}")
        return None


def _generate_gemini_sync(api_key: str, prompt: str, variant: ImageVariant,
                          include_bolt: bool, include_hosts: bool) -> Optional[bytes]:
    """Synchronous Gemini image generation - runs in thread pool

    All PIL operations and SDK calls happen here to avoid thread-safety issues.
    """
    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    # Load and filter reference images as PIL objects
    references = load_host_references_as_pil()
    filtered_refs = [ref for ref in references
                     if (ref["name"] == "Bolt" and include_bolt) or (ref["name"] != "Bolt" and include_hosts)]

    print(f"  📋 Loaded {len(references)} total references, filtered to {len(filtered_refs)} (Bolt={include_bolt}, Hosts={include_hosts})")

    # Build contents list: prompt first, then reference images
    contents = []

    if filtered_refs:
        # Build concise prompt with character descriptions
        character_intro = ""
        if include_bolt:
            character_intro += "Bolt (the blue cloud robot mascot shown in the first reference image). "
        if include_hosts:
            character_intro += "The four podcast hosts (Jonathan, Justin, Matthew, Ryan) shown in the reference photos. Render them in the same cartoon style as Bolt - simple flat vector illustration with distinctive hair patterns: Jonathan has dark wavy hair and is clean-shaven, Justin is completely bald with gray goatee, Matthew has horseshoe hair pattern (bald on top, hair on sides) with full brown beard, Ryan has wavy golden-brown hair with brown goatee. Use their signature outfit colors: blue, gray, orange, teal."

        final_prompt = f"""An illustration featuring {character_intro}

GENERATION TASK:
{prompt}"""

        # Add prompt, then all reference PIL Images
        contents.append(final_prompt)
        for ref in filtered_refs:
            contents.append(ref["image"])
    else:
        contents.append(prompt)

    # Configure generation with SDK types
    aspect_ratio = "1:1" if variant == ImageVariant.SQUARE else "16:9"
    image_size = "4K"  # Maximum quality (options: "1K", "2K", "4K")

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size
        )
    )

    # Call the model using SDK (blocking call)
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=config,
    )

    # Extract image from response
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'inline_data') and part.inline_data:
            return part.inline_data.data

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
    """Generate image using Google Gemini SDK with optional reference images

    Uses the official google-genai SDK. The blocking SDK call runs in a thread pool
    so multiple variants can execute in parallel.
    """

    print(f"  🎨 Gemini 3 Pro Image (Nano Banana Pro) generating {variant.value} variant...")

    try:
        # Run entire synchronous workflow in thread (PIL + SDK calls)
        image_bytes = await asyncio.to_thread(
            _generate_gemini_sync,
            api_key, prompt, variant, include_bolt, include_hosts
        )

        if image_bytes:
            print(f"  ✓ Gemini {variant.value} generated with SDK")
            return image_bytes
        else:
            logger.error("No image data in Gemini SDK response")
            return None

    except Exception as e:
        logger.error(f"Gemini SDK generation error: {e}")
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
    output_path: Path,
    compact_bar: bool = False
) -> bool:
    """Main post-processing pipeline: load → overlay bar → grouped title → logo → save

    Args:
        compact_bar: Use smaller overlay bar (for comic strip layouts where bottom panels need more room)
    """

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
        if compact_bar:
            bar_height = SQUARE_BAR_HEIGHT_COMPACT if is_square else SOCIAL_BAR_HEIGHT_COMPACT
        else:
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

        # Atomic write with web-optimized JPEG settings
        # Square: quality=85 (high-res podcast platforms)
        # Social: quality=80 (smaller social media)
        # progressive=True for better web loading
        jpeg_quality = 85 if variant == ImageVariant.SQUARE else 80
        atomic_write(output_path, lambda path: img.save(
            path, 'JPEG',
            quality=jpeg_quality,
            optimize=True,
            progressive=True
        ))
        print(f"  ✓ Saved: {output_path}")
        return True

    except (OSError, IOError, ValueError) as e:
        logger.error(f"Post-processing failed: {e}")
        return False


# ============================================================================
# ORCHESTRATION AND WORKFLOW
# ============================================================================

async def process_and_save_async(image_bytes: bytes, episode_num: int, episode_title: str,
                                 variant: ImageVariant, output_path: Path, compact_bar: bool = False) -> bool:
    """Thread-pooled PIL post-processing (CPU-bound work off the event loop)"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        PROCESS_POOL, process_and_save,
        image_bytes, episode_num, episode_title, variant, output_path, compact_bar
    )


async def generate_single_variant(
    session: aiohttp.ClientSession,
    api_key: str,
    provider: Provider,
    concept: str,
    episode_num: int,
    episode_title: str,
    output_dir: Path,
    variant_num: int,
    semaphore: asyncio.Semaphore = None,
    concept_label: str = ""
) -> Tuple[int, Optional[Path], Optional[Path]]:
    """Generate a single variant (both square and social)

    Args:
        concept_label: Optional label to distinguish concepts in filenames (e.g. "c1", "c2")

    Returns tuple of (variant_num, square_path, social_path)
    """

    if semaphore:
        await semaphore.acquire()

    try:
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

        # Detect comic strip concepts for compact overlay bar
        # Check for various comic/multi-panel formats
        concept_lower = concept.strip().lower()
        is_comic_strip = (
            concept_lower.startswith("panel 1") or
            "panel 1:" in concept_lower or
            "panel 2:" in concept_lower or
            "panel 1 " in concept_lower or
            "(1)" in concept_lower and "(2)" in concept_lower or  # (1), (2), (3) format
            "four-panel" in concept_lower or
            "4-panel" in concept_lower or
            "comic strip" in concept_lower
        )
        compact_bar = is_comic_strip

        # Build filename with optional concept label for multi-select runs
        label_part = f"-{concept_label}" if concept_label else ""

        # Post-process square and social variants concurrently in thread pool
        filename = f"{episode_num}-{title_slug}{label_part}-{provider.value}-{variant_num}.jpg"
        output_path = output_dir / filename

        filename_social = f"{episode_num}-{title_slug}{label_part}-social-{provider.value}-{variant_num}.jpg"
        output_path_social = output_dir / filename_social

        square_ok, social_ok = await asyncio.gather(
            process_and_save_async(base_image_bytes, episode_num, episode_title, ImageVariant.SQUARE, output_path, compact_bar),
            process_and_save_async(base_image_bytes, episode_num, episode_title, ImageVariant.SOCIAL, output_path_social, compact_bar),
        )

        square_path = output_path if square_ok else None
        social_path = output_path_social if social_ok else None

        return (variant_num, square_path, social_path)

    finally:
        if semaphore:
            semaphore.release()


async def generate_all_variants(
    session: aiohttp.ClientSession,
    api_key: str,
    provider: Provider,
    concept: str,
    episode_num: int,
    episode_title: str,
    output_dir: Path,
    concept_label: str = ""
) -> Dict[ImageVariant, List[Path]]:
    """Generate 4 different images: 2 with Bolt only, 2 with Bolt + hosts

    Returns both square and social variants for each (8 images total per provider)
    """

    results = {ImageVariant.SQUARE: [], ImageVariant.SOCIAL: []}

    # All 4 variants launch concurrently per provider
    # OpenAI: rate limiter spaces API calls; Gemini: runs in thread pool
    semaphore = asyncio.Semaphore(4)

    # Generate all 4 variants concurrently (with controlled concurrency)
    tasks = [
        generate_single_variant(session, api_key, provider, concept, episode_num, episode_title, output_dir, variant_num, semaphore, concept_label)
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
    providers: List[Provider],
    concept_label: str = ""
) -> Dict[Provider, Dict[ImageVariant, List[Path]]]:
    """Generate 4 images per provider concurrently: 2 with Bolt only, 2 with Bolt + hosts

    Runs all providers in parallel for maximum throughput.
    Total output: 8 square + 8 social images per provider
    """

    print(f"\n🖼️  Generating images (providers in parallel)...")
    print("=" * 70)
    print("Each provider will generate:")
    print("  • 2 images with Bolt only")
    print("  • 2 images with Bolt + all four hosts")
    print("  • Each in both square (3000×3000) and social (1200×630) formats")

    # Use episode-numbered subdirectory
    output_dir = OUTPUT_DIR / str(episode_num)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build list of valid providers with their API keys
    valid_providers = []
    for provider in providers:
        if provider == Provider.OPENAI:
            if not OPENAI_API_KEY:
                print("  ⚠️  OpenAI API key not found, skipping")
                continue
            valid_providers.append((provider, OPENAI_API_KEY))
        elif provider == Provider.GEMINI:
            if not GOOGLE_API_KEY:
                print("  ⚠️  Google API key not found, skipping")
                continue
            valid_providers.append((provider, GOOGLE_API_KEY))

    async with aiohttp.ClientSession() as session:
        async def run_provider(provider: Provider, api_key: str):
            print(f"\n📡 {provider.value.upper()} Generation:")
            print("-" * 70)
            return provider, await generate_all_variants(
                session, api_key, provider, selected_concept,
                episode_num, episode_title, output_dir, concept_label
            )

        tasks = [run_provider(p, k) for p, k in valid_providers]
        results = await asyncio.gather(*tasks)

    return {prov: res for prov, res in results}


def save_concepts(episode_num: int, episode_title: str, concepts: List[Tuple[str, str]], selections: List[Tuple[int, str]]):
    """Save concepts to JSON file for reference (atomic write)

    Args:
        selections: List of (index, concept_text) tuples for selected concepts
    """
    # Convert concepts to dict format for JSON
    concepts_list = [{"concept": concept, "provider": provider} for concept, provider in concepts]

    # Build selection data (backwards compatible: first selection also stored in legacy fields)
    first_index, first_concept = selections[0]
    first_provider = concepts[first_index - 1][1]

    concepts_data = {
        "episode": episode_num,
        "title": episode_title,
        "concepts": concepts_list,
        "selected_index": first_index,
        "selected_concept": first_concept,
        "selected_provider": first_provider,
    }

    # Add multi-select data if more than one concept was selected
    if len(selections) > 1:
        concepts_data["all_selections"] = [
            {
                "index": idx,
                "concept": text,
                "provider": concepts[idx - 1][1]
            }
            for idx, text in selections
        ]

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

    # Validate API keys - need at least one image provider (OpenAI or Gemini)
    # Anthropic is concept-only, so it alone isn't sufficient
    if not OPENAI_API_KEY and not GOOGLE_API_KEY:
        print("❌ Error: No image provider API keys found!")
        print("Please set OPENAI_API_KEY and/or GOOGLE_API_KEY in .env file")
        print("(ANTHROPIC_API_KEY is optional, for concept generation only)")
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
    print(f"Image Provider: {args.provider}")
    concept_providers = []
    if OPENAI_API_KEY:
        concept_providers.append("OpenAI")
    if GOOGLE_API_KEY:
        concept_providers.append("Gemini")
    if ANTHROPIC_API_KEY:
        concept_providers.append("Anthropic")
    print(f"Concept Providers: {', '.join(concept_providers)}")

    # Phase 1: Generate concepts (with regeneration loop)
    if args.skip_concepts:
        # Create dummy concepts for testing (one per lens)
        concepts = [(f"Test concept {i+1} for {args.title} [{lens['label']}]", "Test")
                    for i, lens in enumerate(CREATIVE_LENSES)]
        selected_index = args.concept or 1
        selected_concept, _ = concepts[selected_index - 1]
        selections = [(selected_index, selected_concept)]
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
                selections = [(selected_index, selected_concept)]
                print(f"\n✓ Auto-selected concept {selected_index}: {selected_concept}\n")
                break

            # Present concepts and get user choice (supports multi-select)
            selections, should_regenerate = await present_concepts_and_choose(concepts, args.title)

            # If user wants to regenerate, loop again
            if should_regenerate:
                continue
            else:
                break  # User selected concept(s), exit loop

    # Save concepts
    save_concepts(args.episode, args.title, concepts, selections)

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

    # Generate images for each selected concept
    all_results = {}
    multi = len(selections) > 1
    for sel_idx, (concept_index, selected_concept) in enumerate(selections):
        if multi:
            print(f"\n{'=' * 70}")
            print(f"📌 Generating images for concept {sel_idx + 1}/{len(selections)} (#{concept_index})")
            print(f"   {selected_concept[:100]}{'...' if len(selected_concept) > 100 else ''}")
            print(f"{'=' * 70}")

        # Add concept label to filenames when generating multiple concepts
        concept_label = f"c{concept_index}" if multi else ""

        results = await generate_with_providers(
            args.episode,
            args.title,
            selected_concept,
            providers,
            concept_label
        )
        all_results[concept_index] = (selected_concept, results)

    # Print summary for each concept
    for concept_index, (selected_concept, results) in all_results.items():
        if len(all_results) > 1:
            print(f"\n--- Concept #{concept_index} ---")
        print_summary(args.episode, args.title, selected_concept, results)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
        sys.exit(0)
