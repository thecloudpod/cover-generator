# The Cloud Pod - Cover Image Generator

Automated podcast cover image generator using OpenAI and Google Gemini AI APIs.

## Features

- **6 Creative Concepts**: Each provider (OpenAI and Gemini) generates 3 unique concepts inspired by your episode title (6 total)
- **Generate More Concepts**: Add additional concepts anytime with optional keyword steering to guide creative direction
- **Keyword Steering**: Provide keywords (e.g., "space theme", "retro style", "Bolt-focused") to influence concept generation
- **Dual Provider Support**: Both OpenAI (gpt-image-1) and Google Gemini generate images from your selected concept
- **Automatic Character Variants**: Each provider automatically generates 4 versions:
  - 2 images with Bolt only
  - 2 images with Bolt + all four hosts
- **Multiple Formats**: Generates both square (3000×3000) podcast covers and social media (1200×630) variants
- **Total Output**: 16 images per episode (8 square + 8 social: 4 from OpenAI, 4 from Gemini)
- **Professional Overlay Bar**: Semi-transparent bar at bottom with grouped episode number and title for clean, professional look
- **Character Consistency**: Bolt mascot passed as reference image; hosts described via detailed text prompts (names, hair, beards)
- **Scene Consistency**: Each provider generates one base image, reused for both square and social variants with different layouts
- **Rate Limiting**: Handles API rate limits automatically
- **Creative & Varied**: Concepts use literal interpretations, visual puns, and wordplay based on the episode title
- **Interactive Refinement**: Regenerate all concepts (0), generate more (M), refine specific concepts (R#), or exit (X) at any time
- **Concept Refinement**: Ask AI to adjust a concept (e.g., "make it funnier" or "add more Bolt")
- **Protected Visual Area**: AI generation avoids bottom area reserved for title bar, keeping focal points clear

## Setup

1. **Install Dependencies** (automatic on first run):
   ```bash
   ./run.sh --help
   ```
   This will create a virtual environment and install required packages.

2. **API Keys**: Keys are already configured in `.env` file

## Usage

### Interactive Mode (Recommended)

Simply run the script without arguments and it will prompt you:

```bash
./run.sh
```

You'll be prompted to enter:
- Episode number
- Episode title

### Command-Line Mode

Or provide arguments directly:

```bash
./run.sh --episode 337 --title "Your Episode Title Here"
```

The script will:
1. Generate 6 creative concepts (3 from OpenAI, 3 from Gemini)
2. Present concepts with interactive options:
   - `1-N` = Select a concept
   - `0` = Generate 6 brand new concepts
   - `M` = Generate MORE concepts (you can add 3+ concepts with optional keyword steering)
   - `R#` = Refine a specific concept (e.g., `R3` to refine concept 3)
   - `X` = Exit
3. Both providers automatically generate 4 character variants each:
   - 2 images with Bolt only
   - 2 images with Bolt + all four hosts
4. Each variant generated in both square (3000×3000) and social (1200×630) formats
5. Total output: 16 images (8 square + 8 social)
6. Save all images to `output/` directory

### Advanced Options

**Generate with specific provider only:**
```bash
./run.sh --episode 337 --title "Title" --provider openai
./run.sh --episode 337 --title "Title" --provider gemini
```

**Auto-select concept (skip interaction):**
```bash
./run.sh --episode 337 --title "Title" --concept 3
# Choose 1-6 (first 3 are OpenAI concepts, last 3 are Gemini concepts)
```

**Skip concept generation entirely:**
```bash
./run.sh --episode 337 --title "Title" --skip-concepts
```

## Output Structure

All files are saved to a flat `output/` directory with descriptive filenames:

```
output/
├── 337-we-were-right-gemini-1.jpg          # Gemini variant 1 (Bolt only) - 3000×3000
├── 337-we-were-right-gemini-2.jpg          # Gemini variant 2 (Bolt only) - 3000×3000
├── 337-we-were-right-gemini-3.jpg          # Gemini variant 3 (Bolt + hosts) - 3000×3000
├── 337-we-were-right-gemini-4.jpg          # Gemini variant 4 (Bolt + hosts) - 3000×3000
├── 337-we-were-right-social-gemini-1.jpg   # Gemini variant 1 social - 1200×630
├── 337-we-were-right-social-gemini-2.jpg   # Gemini variant 2 social - 1200×630
├── 337-we-were-right-social-gemini-3.jpg   # Gemini variant 3 social - 1200×630
├── 337-we-were-right-social-gemini-4.jpg   # Gemini variant 4 social - 1200×630
├── 337-we-were-right-openai-1.jpg          # OpenAI variant 1 (Bolt only) - 3000×3000
├── 337-we-were-right-openai-2.jpg          # OpenAI variant 2 (Bolt only) - 3000×3000
├── 337-we-were-right-openai-3.jpg          # OpenAI variant 3 (Bolt + hosts) - 3000×3000
├── 337-we-were-right-openai-4.jpg          # OpenAI variant 4 (Bolt + hosts) - 3000×3000
├── 337-we-were-right-social-openai-1.jpg   # OpenAI variant 1 social - 1200×630
├── 337-we-were-right-social-openai-2.jpg   # OpenAI variant 2 social - 1200×630
├── 337-we-were-right-social-openai-3.jpg   # OpenAI variant 3 social - 1200×630
├── 337-we-were-right-social-openai-4.jpg   # OpenAI variant 4 social - 1200×630
└── 337-concepts.json                       # Saved concepts for reference
```

**Filename Format:**
- Square (podcast): `{episode}-{title-keywords}-{provider}-{1-4}.jpg`
- Social media: `{episode}-{title-keywords}-social-{provider}-{1-4}.jpg`
- Concepts: `{episode}-concepts.json`

**Variant Guide:**
- Variants 1-2: Bolt only (two different interpretations)
- Variants 3-4: Bolt + all four hosts (two different interpretations)

## Example

### Interactive Mode
```bash
./run.sh

Episode number: 337
Episode title: We Were Right (Mostly), 2026: The New Prophecies
```

### Command-Line Mode
```bash
./run.sh --episode 337 --title "We Were Right (Mostly), 2026: The New Prophecies"
```

### Using "Generate More" with Keyword Steering

After seeing the initial 6 concepts, you can generate more with keyword guidance:

```
Commands: [1-6] = Select concept | 0 = Generate 6 new concepts | M = Generate MORE concepts | R1-R6 = Refine concept | X = Exit

Your choice: M

Optional keywords to steer concepts (or press Enter to skip): space theme, futuristic

How many additional concepts? (default: 3): 3

🎨 Generating 3 more concepts...
   Keywords: space theme, futuristic
```

The AI will generate additional concepts that incorporate your keywords while still using literal interpretation of the episode title.

### Generated Output

Generates 16 high-quality images per episode (4 character variants per provider):

**Gemini (8 images):**
- Variants 1-2: Bolt only (2 interpretations × 2 formats = 4 images)
- Variants 3-4: Bolt + hosts (2 interpretations × 2 formats = 4 images)

**OpenAI (8 images):**
- Variants 1-2: Bolt only (2 interpretations × 2 formats = 4 images)
- Variants 3-4: Bolt + hosts (2 interpretations × 2 formats = 4 images)

Each image includes:
- Creative visualization inspired by episode title
- Semi-transparent overlay bar at bottom
- Episode number and title (grouped together)
- The Cloud Pod logo
- Professional typography on dark background

## Technical Details

- **Image Dimensions**:
  - Square: 3000×3000 (podcast platforms)
  - Social: 1200×630 (Facebook/LinkedIn Open Graph)
- **Typography**: Helvetica with clean white text on semi-transparent overlay bar
- **Overlay Bar**: Semi-transparent black bar (650px square, 160px social) at bottom for title/logo
- **Logo**: Positioned in overlay bar (bottom-right corner)
- **Character References**:
  - Bolt: Reference image (bolt.png) passed to both providers
  - Hosts: Detailed text descriptions (Jonathan: short dark hair/clean-shaven, Justin: bald/gray beard, Matthew: balding/brown beard, Ryan: wavy hair/brown beard)
- **Reference Image Support**:
  - OpenAI: Uses `images.edit()` endpoint with `input_fidelity: high` for Bolt reference
  - Gemini: Native reference image support via inline_data for Bolt reference
- **Smart Aspect Ratio Handling**: Social images use blurred letterbox backgrounds to avoid distortion when converting square to landscape
- **AI Models**:
  - OpenAI: gpt-image-1 via edit endpoint with high input fidelity, GPT-4 (concepts)
  - Gemini: gemini-2.5-flash-image with reference images, gemini-2.5-flash (concepts)

## Files

- `generate_podcast_cover.py` - Main generation script
- `run.sh` - Shell wrapper with venv management
- `requirements.txt` - Python dependencies
- `.env` - API keys (already configured)
- `.gitignore` - Excludes sensitive files

## Future Enhancements

- Host image compositing
- Additional social formats (Instagram square, YouTube thumbnail)
- Batch generation from CSV
- Web interface
