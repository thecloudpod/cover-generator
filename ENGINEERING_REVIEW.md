# Principal Engineer Review - Code Quality & Resilience

**Review Date:** 2026-01-09
**Lines of Code:** 1,459
**Functions/Classes:** 26
**Severity Levels:** 🔴 Critical | 🟡 Medium | 🟢 Low

---

## **CRITICAL ISSUES** 🔴

### **1. Bare Exception Handlers - Dangerous Error Swallowing**

**Location:** Lines 827, 832 (get_font), plus 9x "except Exception as e"

**Problem:**
```python
try:
    font = ImageFont.truetype(FONT_PATH, size, index=1)
    return font
except:  # 🔴 Catches EVERYTHING including KeyboardInterrupt, SystemExit
    try:
        font = ImageFont.truetype(FONT_PATH, size)
        return font
    except:  # 🔴 Again!
        return ImageFont.load_default()
```

**Impact:**
- `KeyboardInterrupt` (Ctrl+C) won't work during font loading
- Silent failures hide real problems
- "except Exception as e" still too broad (catches MemoryError, etc.)

**Fix:**
```python
try:
    font = ImageFont.truetype(FONT_PATH, size, index=1)
    return font
except (OSError, IOError) as e:  # Specific font file errors
    try:
        font = ImageFont.truetype(FONT_PATH, size)
        return font
    except (OSError, IOError):
        print(f"  ⚠️  Font loading failed, using default: {e}")
        return ImageFont.load_default()
```

**Action Required:**
- Replace all 11 bare exception handlers with specific exceptions
- Never catch `Exception` without re-raising SystemExit/KeyboardInterrupt

---

### **2. No File Size Validation Before Base64 Encoding**

**Location:** `load_bolt_reference()` (line 99), `load_host_references()` (line 112)

**Problem:**
```python
with open(BOLT_PATH, 'rb') as f:
    image_data = f.read()  # 🔴 Could be 500MB, loads entirely into memory
    return base64.b64encode(image_data).decode('utf-8')
```

**Impact:**
- Malicious/accidental huge file causes MemoryError
- base64 encoding makes it 33% larger (500MB → 666MB in memory)
- Multiple files → OOM kill

**Edge Cases:**
- User accidentally puts 4K video file in Hosts/ folder
- Corrupt image file with incorrect size header
- Network-mounted Hosts/ folder with slow I/O

**Fix:**
```python
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit

def load_bolt_reference() -> Optional[str]:
    try:
        if not BOLT_PATH.exists():
            return None

        file_size = BOLT_PATH.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            print(f"  ⚠️  Bolt image too large ({file_size / 1024 / 1024:.1f}MB), skipping")
            return None

        with open(BOLT_PATH, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
    except (OSError, IOError) as e:
        print(f"  ⚠️  Could not load Bolt reference: {e}")
        return None
```

**Action Required:**
- Add MAX_IMAGE_SIZE constant (10MB recommended)
- Validate size before reading
- Add test with 100MB dummy file

---

### **3. No Retry Logic for API Calls**

**Location:** All API calls (OpenAI, Gemini text, Gemini image)

**Problem:**
- Transient network errors cause complete failure
- Rate limit errors don't retry with backoff
- Timeout errors abandon generation immediately

**Current Behavior:**
```python
async with session.post(endpoint, json=payload, timeout=120) as response:
    if response.status == 200:
        return data
    else:
        print("❌ Generation failed")  # 🔴 Game over!
        return None
```

**Edge Cases:**
- Gemini succeeds, OpenAI has momentary network glitch → only 2 images
- API returns 503 (temporary unavailable) → fail instead of retry
- DNS resolution fails momentarily → fail instead of retry

**Fix (with exponential backoff):**
```python
async def api_call_with_retry(
    session, method, url, max_retries=3, initial_delay=1, **kwargs
):
    """Retry API calls with exponential backoff"""
    last_error = None

    for attempt in range(max_retries):
        try:
            async with session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limit
                    delay = initial_delay * (2 ** attempt)
                    print(f"  ⏳ Rate limited, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                elif response.status >= 500:  # Server error
                    delay = initial_delay * (2 ** attempt)
                    print(f"  ⚠️  Server error {response.status}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                else:  # Client error (400-499 except 429)
                    return None  # Don't retry client errors
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e
            delay = initial_delay * (2 ** attempt)
            print(f"  ⚠️  Network error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)

    print(f"  ❌ API call failed after {max_retries} retries: {last_error}")
    return None
```

**Action Required:**
- Implement retry wrapper
- Apply to all API calls
- Add --no-retry flag for testing

---

### **4. Mixed Sync/Async Patterns - Technical Debt**

**Location:** `refine_concept()` uses `requests`, rest uses `aiohttp`

**Problem:**
```python
def refine_concept(...):  # Synchronous function
    response = requests.post(...)  # Blocks entire event loop!
```

Called from:
```python
async def main():
    ...
    refined = refine_concept(...)  # 🔴 Blocks all async operations!
```

**Impact:**
- During refinement, entire program freezes
- Can't cancel refinement with Ctrl+C reliably
- If refinement takes 30s, user sees no progress

**Fix:**
```python
async def refine_concept(...):  # Make async
    async with aiohttp.ClientSession() as session:
        async with session.post(...) as response:
            ...
```

**Action Required:**
- Convert refine_concept to async
- Remove `requests` dependency
- Update call sites

---

## **MEDIUM PRIORITY ISSUES** 🟡

### **5. No Input Validation on User-Provided Data**

**Location:** `parse_args()`, `get_interactive_input()`

**Problem:**
```python
args.episode = int(episode_input)  # 🟡 What if episode_input = "999999999999"?
args.title = title_input.strip()   # 🟡 What if title is 10,000 characters?
```

**Edge Cases:**
- Episode 0, negative episode, episode > 10000
- Title with only emojis or special chars
- Title with path traversal: "../../etc/passwd"
- Title longer than filesystem limits (255 chars)

**Fix:**
```python
# Constants
MIN_EPISODE = 1
MAX_EPISODE = 9999
MAX_TITLE_LENGTH = 200

def validate_episode(episode: int) -> bool:
    if not MIN_EPISODE <= episode <= MAX_EPISODE:
        print(f"❌ Episode must be between {MIN_EPISODE} and {MAX_EPISODE}")
        return False
    return True

def validate_title(title: str) -> bool:
    if not title or len(title.strip()) == 0:
        print("❌ Title cannot be empty")
        return False
    if len(title) > MAX_TITLE_LENGTH:
        print(f"❌ Title too long (max {MAX_TITLE_LENGTH} chars)")
        return False
    # Check for path traversal
    if '..' in title or '/' in title or '\\' in title:
        print("❌ Title contains invalid characters")
        return False
    return True
```

**Action Required:**
- Add validation constants
- Validate in get_interactive_input
- Add unit tests for edge cases

---

### **6. slugify_title() Can Produce Empty Strings**

**Location:** Line 81

**Problem:**
```python
def slugify_title(title: str, max_words: int = 3) -> str:
    title_clean = re.sub(r'[^\w\s-]', '', title.lower())
    words = title_clean.split()[:max_words]
    slug = '-'.join(words)
    return slug  # 🟡 Could be empty string!
```

**Edge Cases:**
- Title: "☁️ 🚀 💻" (all emojis) → slug = "" → filename = "337--openai.jpg"
- Title: "... ... ..." (only punctuation) → slug = ""
- Title: "a" (one letter) → slug = "a" → potential collisions

**Fix:**
```python
def slugify_title(title: str, max_words: int = 3) -> str:
    title_clean = re.sub(r'[^\w\s-]', '', title.lower())
    words = title_clean.split()[:max_words]

    if not words:  # Handle edge case
        # Fallback to sanitized version of first 20 chars
        slug = re.sub(r'[^\w-]', '', title[:20].lower())
        if not slug:
            return "untitled"  # Last resort
    else:
        slug = '-'.join(words)

    return slug
```

**Action Required:**
- Add fallback logic
- Add test: test_slugify_emoji_only()
- Add test: test_slugify_punctuation_only()

---

### **7. No Concept File Corruption Handling**

**Location:** `save_concepts()` line 1202

**Problem:**
```python
with open(concepts_file, 'w') as f:
    json.dump(concepts_data, f, indent=2)  # 🟡 What if disk full? Power loss?
```

**Edge Cases:**
- Disk full during write → corrupt JSON
- Power loss mid-write → half-written file
- Next run reads corrupt file → crash

**Fix (atomic write):**
```python
import tempfile
import shutil

def save_concepts_atomic(episode_num, title, concepts, selected_index):
    concepts_data = {...}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    concepts_file = OUTPUT_DIR / f"{episode_num}-concepts.json"
    temp_file = concepts_file.with_suffix('.json.tmp')

    try:
        # Write to temp file first
        with open(temp_file, 'w') as f:
            json.dump(concepts_data, f, indent=2)

        # Atomic rename (POSIX guarantees atomicity)
        shutil.move(str(temp_file), str(concepts_file))
        print(f"  ✓ Concepts saved to: {concepts_file}")
    except (OSError, IOError) as e:
        print(f"  ❌ Failed to save concepts: {e}")
        if temp_file.exists():
            temp_file.unlink()
```

**Action Required:**
- Implement atomic write
- Add disk space check before writing
- Test with full disk

---

### **8. Font Loading Fragility**

**Location:** Line 66, 821-834

**Problem:**
```python
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"  # 🟡 macOS only!
```

**Edge Cases:**
- Linux: Helvetica not at this path
- Windows: Path uses backslashes, different fonts
- Docker: No fonts installed
- Custom macOS: User removed Helvetica

**Fix:**
```python
import platform

def get_font_path():
    """Get platform-appropriate font path"""
    system = platform.system()

    if system == "Darwin":  # macOS
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ]
    elif system == "Windows":
        candidates = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]

    for path in candidates:
        if Path(path).exists():
            return path

    return None  # Will use default font

FONT_PATH = get_font_path()
```

**Action Required:**
- Make font path cross-platform
- Add fallback font list
- Document font requirements in README

---

## **LOW PRIORITY / CLEANUP** 🟢

### **9. Magic Numbers and Constants**

**Issues:**
- Line 477: `max_tokens=500` hardcoded (concept refinement)
- Line 362: `max_tokens=300` hardcoded (OpenAI concept generation)
- Line 955: Blur radius `50` hardcoded
- Line 956: Brightness `0.5` hardcoded
- Line 957: Color `0.7` hardcoded

**Fix:** Extract to configuration section
```python
# Text Generation Config
CONCEPT_GENERATION_TOKENS = 300
CONCEPT_REFINEMENT_TOKENS = 500

# Image Processing Config
LETTERBOX_BLUR_RADIUS = 50
LETTERBOX_BRIGHTNESS = 0.5
LETTERBOX_SATURATION = 0.7
```

---

### **10. No Logging - Only Print Statements**

**Problem:** Can't debug production issues, no log levels

**Fix:** Add proper logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('cover-generator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info("Starting generation for episode %d", episode_num)
logger.warning("Bolt reference image not found")
logger.error("OpenAI API call failed: %s", error)
```

**Benefits:**
- Can review logs after failure
- Can set log level (DEBUG vs INFO vs ERROR)
- Timestamps for performance analysis

---

### **11. No Progress Indication for Long Operations**

**Problem:** Image generation can take 2+ minutes, user sees nothing

**Fix:** Add progress indicators
```python
from tqdm import tqdm  # or custom progress

async def generate_with_providers(...):
    print(f"\n🖼️  Generating images...")

    with tqdm(total=len(providers) * 2, desc="Generating") as pbar:
        for provider in providers:
            # ... generate square
            pbar.update(1)
            # ... generate social
            pbar.update(1)
```

---

### **12. Monolithic File Structure**

**Current:** 1,459 lines, single file

**Issues:**
- Hard to navigate
- Hard to test individual functions
- Hard to reuse code

**Suggested Structure:**
```
cover-generator/
├── core/
│   ├── __init__.py
│   ├── config.py          # All constants and config
│   ├── prompts.py         # Prompt building functions
│   ├── api_clients.py     # OpenAI and Gemini clients
│   └── image_processing.py # PIL operations
├── utils/
│   ├── __init__.py
│   ├── validation.py      # Input validation
│   └── file_utils.py      # File operations
├── generate_podcast_cover.py  # Main entry point (200 lines)
├── tests/
│   ├── test_validation.py
│   ├── test_slugify.py
│   └── test_image_processing.py
└── requirements.txt
```

---

### **13. No Unit Tests**

**Risk:** Every change could break something

**Minimum Test Suite:**
```python
# tests/test_validation.py
def test_slugify_normal():
    assert slugify_title("Hello World Test") == "hello-world-test"

def test_slugify_emoji_only():
    assert slugify_title("☁️ 🚀 💻") == "untitled"

def test_slugify_punctuation():
    assert slugify_title("... ... ...") == "untitled"

def test_episode_validation():
    assert validate_episode(337) == True
    assert validate_episode(0) == False
    assert validate_episode(10001) == False
```

---

### **14. Hardcoded URLs and Endpoints**

**Problem:** Can't test against staging, can't switch API versions

**Fix:**
```python
class Config:
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

    @property
    def openai_chat_endpoint(self):
        return f"{self.OPENAI_BASE_URL}/chat/completions"
```

---

## **PERFORMANCE CONSIDERATIONS**

### **15. Sequential Provider Execution**

**Current:** Gemini → wait → OpenAI → wait

**Potential:** Run both concurrently
```python
async def generate_with_providers(...):
    # Generate both providers in parallel
    tasks = []
    for provider in providers:
        tasks.append(generate_provider(provider, ...))

    results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Tradeoff:** Both APIs hit at once (may hit rate limits faster)

---

### **16. Image Processing Not Optimized**

**Location:** Letterbox creation with blur

**Current:** 50px Gaussian blur on 1200x630 is expensive

**Optimization:**
- Resize down before blur, resize back up
- Or use faster box blur

---

## **SECURITY CONSIDERATIONS**

### **17. API Keys in .env with No Validation**

**Problem:** Empty/invalid keys not detected until API call fails

**Fix:**
```python
def validate_api_keys():
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-"):
        print("⚠️  OpenAI API key format looks invalid")

    if GOOGLE_API_KEY and len(GOOGLE_API_KEY) < 30:
        print("⚠️  Google API key looks too short")
```

---

### **18. No Rate Limit Tracking**

**Problem:** Could exhaust API quotas without warning

**Fix:** Track API usage
```python
class APIUsageTracker:
    def __init__(self):
        self.openai_calls = 0
        self.gemini_calls = 0

    def log_call(self, provider: str):
        if provider == "openai":
            self.openai_calls += 1
            if self.openai_calls > 50:  # Warning threshold
                print("⚠️  High OpenAI usage today")
```

---

## **IMPLEMENTATION PRIORITY**

### **Sprint 1 (Critical - Week 1):**
1. Fix bare exception handlers → Specific exceptions
2. Add file size validation → Prevent OOM
3. Implement retry logic → Improve reliability
4. Convert refine_concept to async → Fix blocking

### **Sprint 2 (Medium - Week 2):**
5. Add input validation → Prevent edge cases
6. Fix slugify edge cases → Prevent empty filenames
7. Implement atomic file writes → Prevent corruption
8. Cross-platform font loading → Linux/Windows support

### **Sprint 3 (Low - Week 3+):**
9. Extract constants to config section
10. Add logging framework
11. Add progress indicators
12. Write unit test suite

### **Backlog (Future):**
13. Refactor into modules
14. Add performance optimizations
15. Implement usage tracking

---

## **TESTING RECOMMENDATIONS**

### **Edge Cases to Test:**
```bash
# Invalid inputs
./run.sh --episode 0 --title ""
./run.sh --episode 99999 --title "$(python -c 'print("A" * 1000)')"
./run.sh --episode 337 --title "☁️ 🚀 💻"

# Missing files
rm Logo/smallsquare.png && ./run.sh --episode 337 --title "Test"
rm Hosts/bolt.png && ./run.sh --episode 337 --title "Test with Bolt"

# Disk full simulation (Linux)
dd if=/dev/zero of=test.img bs=1M count=10
mkfs.ext4 test.img
mount -o loop test.img /mnt/test
cd /mnt/test && ./run.sh ...  # Should handle gracefully

# Network failures
# Use toxiproxy or similar to simulate timeouts, 5xx errors

# Large files
cp /dev/urandom Hosts/huge.png  # Create 100MB file
# Should reject with warning, not crash
```

---

## **SUMMARY**

**Code Quality:** 6/10
- Works for happy path
- Fragile on edge cases
- Limited error recovery

**Recommended Actions:**
1. 🔴 Fix 4 critical issues (Weeks 1-2)
2. 🟡 Address 4 medium issues (Week 3)
3. 🟢 Cleanup & testing (Week 4+)

**Estimated Effort:** 3-4 weeks for hardening

**Risk if not addressed:**
- Production failures from transient errors
- Memory issues with large files
- User frustration from missing validation
- Platform incompatibility (Linux/Windows)
