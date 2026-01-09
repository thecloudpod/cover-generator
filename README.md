# The Cloud Pod - Cover Image Generator

Automated podcast cover image generator using OpenAI and Google Gemini AI APIs.

## Features

- **6 Creative Concepts**: Each provider (OpenAI and Gemini) generates 3 unique concepts inspired by your episode title (6 total)
- **Generate More Concepts**: Add additional concepts anytime with optional keyword steering to guide creative direction
- **Keyword Steering**: Provide keywords (e.g., "space theme", "retro style", "Bolt-focused") to influence concept generation
- **Dual Provider Support**: Both OpenAI (gpt-image-1) and Google Gemini generate images from your selected concept
- **Multiple Image Variants**: Each provider generates 2 different interpretations of the selected concept (4 square + 4 social = 8 images total)
- **Multiple Formats**: Generates both square (3000×3000) podcast covers and social media (1200×630) variants
- **Professional Overlay Bar**: Semi-transparent bar at bottom with grouped episode number and title for clean, professional look
- **Character Consistency**: Bolt mascot passed as reference image; hosts described via detailed text prompts (names, hair, beards)
- **Scene Consistency**: Each provider generates one base image, reused for both square and social variants with different layouts
- **Rate Limiting**: Handles API rate limits automatically
- **Creative & Varied**: Concepts use literal interpretations, visual puns, and wordplay based on the episode title
- **Character Integration**: Bolt (mascot) and silhouetted hosts automatically used when they enhance the concept
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
3. Both providers generate images based on your selected/refined concept
4. Generate 4 images total per provider (2 variants × 2 formats = 8 total)
5. Save them to `output/337/`

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
├── 337-we-were-right-gemini.jpg          # 3000×3000 podcast cover (Gemini)
├── 337-we-were-right-social-gemini.jpg   # 1200×630 social media (Gemini)
├── 337-we-were-right-openai.jpg          # 3000×3000 podcast cover (OpenAI)
├── 337-we-were-right-social-openai.jpg   # 1200×630 social media (OpenAI)
└── 337-concepts.json                     # Saved concepts for reference
```

**Filename Format:**
- Square (podcast): `{episode}-{title-keywords}-{provider}.jpg`
- Social media: `{episode}-{title-keywords}-social-{provider}.jpg`
- Concepts: `{episode}-concepts.json`

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

Generates 8 high-quality images per episode (2 interpretations per provider):
- `337-we-were-right-gemini-1.jpg` - First Gemini interpretation (3000×3000)
- `337-we-were-right-gemini-2.jpg` - Second Gemini interpretation (3000×3000)
- `337-we-were-right-social-gemini-1.jpg` - First Gemini social variant (1200×630)
- `337-we-were-right-social-gemini-2.jpg` - Second Gemini social variant (1200×630)
- `337-we-were-right-openai-1.jpg` - First OpenAI interpretation (3000×3000)
- `337-we-were-right-openai-2.jpg` - Second OpenAI interpretation (3000×3000)
- `337-we-were-right-social-openai-1.jpg` - First OpenAI social variant (1200×630)
- `337-we-were-right-social-openai-2.jpg` - Second OpenAI social variant (1200×630)

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
