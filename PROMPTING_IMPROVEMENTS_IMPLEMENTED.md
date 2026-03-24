# Provider-Specific Prompting Improvements - Implementation Summary

## Overview

Refactored `build_image_prompt()` to follow official prompting guides from OpenAI and Google Gemini, with provider-specific prompt structures optimized for cartoon/illustration style.

## Key Changes

### 1. Modular Style Components

**Before:** Single monolithic `BASE_STYLE_PROMPT` string
**After:** Separate reusable components for flexible composition

```python
UNIVERSE_CONTEXT       # Podcast context and brand character notes
ILLUSTRATION_STYLE     # Visual medium, materials & texture
CHARACTER_DESIGN       # Design principles for characters
COLOR_PALETTE          # Color scheme with hex codes
COMPOSITION_BASE       # Composition and framing rules
HUMOR_STORYTELLING     # Humor and storytelling guidelines
```

**Benefits:**
- Easier to maintain and update individual sections
- Provider-specific prompts can include/exclude components as needed
- Better code organization

### 2. Provider-Specific Prompt Structures

#### OpenAI Structure (per OpenAI Cookbook)
```
Background/Scene → Subject → Details → Constraints
```

**Implementation:**
- **Background/Scene:** Universe context + concept description
- **Subject Details:** Illustration style, character designs, colors
- **Composition & Framing:** Dimension guidance, viewpoint, element placement
- **Constraints:** PRESERVE/INCLUDE ONLY/CHANGE ONLY directives

**Key OpenAI Optimizations:**
- Explicit preservation constraints (prevent drift during iterations)
- "INCLUDE ONLY" constraints to avoid adding unwanted elements
- Material descriptions for cartoon style (matte, smooth gradients, vector edges)
- Avoidance of photographic terms (for illustration style)

#### Gemini Structure (per Google Guide)
```
Subject → Action → Location → Composition → Style
```

**Implementation:**
- **Subject & Action:** Character descriptions + concept
- **Location & Setting:** Universe context
- **Composition & Camera:** Dimension guidance + framing terms
- **Visual Style:** Illustration style, design principles, colors

**Key Gemini Optimizations:**
- Camera/lighting terminology (medium shot, straight-on angle, directional lighting)
- Reference image consistency emphasis
- Geometric shape language
- Silhouette testing reminder
- Crowd scene duplication guidance

### 3. Universal Improvements (Both Providers)

#### Text Rendering with Quotes
**Before:** `"A sign that says Cloud Zone"`
**After:** `'A sign displaying "CLOUD ZONE" in bold sans-serif font'`

Both guides recommend enclosing literal text in quotes for better accuracy.

#### Condensed Character Descriptions
**Before:** 150+ lines of detailed character specifications
**After:** Compact descriptions preserving all essential information

- Bolt: 6 lines (was ~20)
- Hosts: 8 lines total (was ~110)

**Benefits:**
- Reduced token usage
- Faster processing
- Core information preserved (PRESERVE statements for OpenAI)

#### Cartoon-Appropriate Materials
Added material descriptions suitable for flat vector illustration:
- "Smooth matte solid colors with soft gradient shading"
- "Cotton-like soft texture, simplified geometric shape"
- "Clean vector edges, no photorealistic texture detail"

**Rationale:** OpenAI guide specifically recommends describing materials/textures for illustration styles (not just photorealistic renders).

#### Positive Framing + Constraints
**Before:** "No graphic design overlays", "Do not add extra elements"
**After (OpenAI):** "CONSTRAINTS: INCLUDE ONLY elements described in concept"
**After (Gemini):** Focus on what to include, with minimal negatives

### 4. Preservation System (OpenAI Emphasis)

Added explicit PRESERVE statements for character consistency:

```python
PRESERVE:
- Bolt's blue color (#0066FF), lightning bolt chest icon orientation
- Host hair patterns, facial hair styles, signature outfit colors
- Character colors, core designs, signature visual elements
```

**Why:** OpenAI cookbook emphasizes stating what must remain unchanged to prevent drift across iterations and multi-image generations.

### 5. Composition Control

Both providers now get specific composition guidance:

**OpenAI:**
- Viewpoint: "Straight-on or slight low-angle for approachable feel"
- Element placement: "Center focal point in upper 75% of frame"

**Gemini:**
- Framing: "Medium shot, straight-on or slight low-angle"
- Focus: "Everything sharp and clear (deep focus)"
- Lighting: "Soft directional light from upper left, gentle shadows for depth"

**Note:** Kept cartoon-appropriate (no f-stops or lens specifications), but added framing terminology both providers respond to.

## What Was NOT Changed

### Preserved Systems
- ✅ Creative lens system (pop culture parody, hero shot, etc.)
- ✅ Concept generation prompts (still use BASE_STYLE_PROMPT)
- ✅ Comic strip detection and layout handling
- ✅ Character selection logic (Bolt/hosts optional)
- ✅ Dimension guidance (upper 75% rule for text overlay)
- ✅ Duplication guidance for crowd scenes

### Avoided Additions
- ❌ Photography terms (f-stop, focal length, aperture) - not appropriate for cartoons
- ❌ Complex lighting setups (3-point studio lighting) - cartoons use simple shading
- ❌ Film stock terminology - not relevant to flat vector illustration
- ❌ Generic quality descriptors ("8K", "ultra HD") - both guides say to be specific instead

## Testing Recommendations

### Before/After Comparison
Test the same episode concept with both old and new prompts:

1. **Character consistency:** Do Bolt and hosts look more consistent across variants?
2. **Text rendering:** If concepts include text, does quoted syntax improve legibility?
3. **Composition:** Is framing guidance being followed better?
4. **Style adherence:** Does the flat vector style remain consistent?

### Provider Comparison
Generate the same concept with both OpenAI and Gemini:

1. **Quality:** Any noticeable quality differences?
2. **Consistency:** Which provider maintains character designs better?
3. **Text handling:** Which handles text in images better?
4. **Speed:** Cost/latency differences?

### Edge Cases
1. **Comic strips:** Do 4-panel layouts still work correctly?
2. **Crowd scenes:** Are duplicated hosts properly varied?
3. **Character-free concepts:** Do abstract/environmental concepts work?
4. **Text-heavy concepts:** Signs, labels, speech bubbles rendering correctly?

## Files Modified

- `generate_podcast_cover.py` (lines 495-951)
  - Broke `BASE_STYLE_PROMPT` into modular components
  - Completely rewrote `build_image_prompt()` function
  - Added provider-specific prompt building logic

## Backward Compatibility

✅ **Fully backward compatible:**
- Concept generation still uses `BASE_STYLE_PROMPT` (reconstructed from components)
- Function signature unchanged: `build_image_prompt(concept, variant, provider, include_bolt, include_hosts)`
- All existing logic preserved (comic strip detection, character selection, etc.)

## Next Steps

1. **Test generation** with a few episodes to validate improvements
2. **Monitor quality** - compare old vs new outputs
3. **Iterate if needed** - fine-tune based on actual results
4. **Document findings** - which improvements had the biggest impact?

## References

- **OpenAI:** https://developers.openai.com/cookbook/examples/multimodal/image-gen-1.5-prompting_guide/
- **Google:** "Ultimate prompting guide for Nano Banana" (Gemini image models)
- **Key insight:** Different providers prefer different prompt structures; optimize for each
