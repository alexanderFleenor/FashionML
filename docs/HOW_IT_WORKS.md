# How the Outfit Recommendation System Works

## How the Pieces Fit Together

This document explains how this outfit recommendation system works from start to finish. If you've ever wondered how computers can learn to match clothes, this is for you.

---

## The Big Picture

Imagine you have a closet full of clothes and you're trying to figure out what to wear. You naturally consider things like:
- What type of clothing is each piece? (shirt, pants, shoes, etc.)
- Do these colors go together?
- Does this outfit "look right" as a whole?

This system does exactly the same thing, but using artificial intelligence. It looks at photos of your clothes, figures out what each item is, analyzes the colors, and then uses patterns it learned from thousands of real outfits to suggest combinations that work well together.

---

## The Four Main Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR CLOSET (Photos of Clothes)                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: RECOGNITION                                               │
│  "What type of clothing is this?"                                   │
│  The computer identifies: top, bottom, dress, shoes, or accessory   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: ATTRIBUTE EXTRACTION                                      │
│  "What does this look like?"                                        │
│  Extracts colors, patterns, and visual characteristics              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: COMPATIBILITY SCORING                                     │
│  "Do these items go well together?"                                 │
│  Uses AI trained on real outfits to score how well items match      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: OUTFIT GENERATION                                         │
│  "What are the best outfit combinations?"                           │
│  Combines items and ranks the best options                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED OUTFITS + EXPLANATIONS               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Garment Recognition

### What It Does
When you upload a photo of a clothing item, the first thing the system needs to know is: **what type of clothing is this?**

The system classifies items into 5 categories:
- **Tops** (shirts, blouses, sweaters, jackets)
- **Bottoms** (pants, shorts, skirts)
- **Dresses** (full outfits that cover top and bottom)
- **Shoes** (sneakers, heels, boots, sandals)
- **Accessories** (bags, hats, jewelry, scarves)

### How It Works: Neural Networks

The recognition is done by a **neural network**, which learns patterns from labeled examples. In this case, the model was trained on clothing images like:
- "This is a top"
- "This is a pair of shoes"
- "This is a dress"

After training, the network can recognize the visual patterns that distinguish each category.

**The specific network used here is EfficientNet-B0.** It starts from ImageNet weights and is fine-tuned for clothing categories.

### Visual Features

Besides classifying the item, the network also extracts **visual features**: a list of 1,280 numbers that represent the image in a form the compatibility model can use.

---

## Stage 2: Attribute Extraction

### Color Extraction

Colors are crucial for outfit matching. The system extracts the **dominant colors** from each clothing item.

Here's how it works:

1. **Remove the background**: The system detects edges in the image to separate the clothing item from the background (usually white or gray in product photos).

2. **Filter out skin tones**: If a person is wearing the item, skin pixels should not count as clothing colors.

3. **Find the main colors**: Using a technique called **K-means clustering**, the system groups all the pixels by color and finds the 5 most common colors in the garment.

4. **Use the LAB color space**: Instead of regular RGB (Red-Green-Blue), the system uses a color space called LAB which more closely matches how humans perceive color differences.

**Example output:**
```
Item: Navy Blue Sweater
Dominant colors:
  1. Navy blue (45%)
  2. Dark blue (30%)
  3. Light blue accent (15%)
  4. White trim (10%)
```

### Color Harmony Analysis

The system also understands **color theory** - the artistic principles about which colors work well together:

- **Analogous colors**: Colors next to each other on the color wheel (blue + green + teal)
- **Complementary colors**: Colors opposite each other (blue + orange)
- **Neutral combinations**: Black, white, gray, beige - these go with everything
- **Triadic colors**: Three colors equally spaced on the color wheel

This helps the system understand *why* certain combinations work.

### Multi-Color Pattern Classification

The system now automatically classifies garments by their color pattern:

| Pattern | Criteria | Example |
|---------|----------|---------|
| **Solid** | Primary color >= 85% | A plain navy blue shirt |
| **Two-tone** | Primary 50-85% + significant secondary | A white shirt with navy stripes |
| **Multi-color** | Primary < 50% or 3+ colors | A colorful Hawaiian shirt |

**Why this matters:**

Fashion stylists know that mixing patterns is tricky. A solid navy shirt pairs well with almost anything, but a busy patterned shirt requires more care. The system uses this classification to:

1. **Score outfit compatibility** - All-solid outfits get a "classic and safe" bonus
2. **Detect pattern clashes** - Warns when two multi-color items might clash
3. **Find coordinated outfits** - Boosts scores when a solid item matches a color in a patterned piece

**Example output:**
```
Item: Striped Oxford Shirt
Color Pattern: TWO-TONE
  Primary: white (55%)
  Secondary: navy (40%)
  is_solid(): False
  is_multicolor(): False
  is_two_tone(): True
  Color Summary: white and navy
```

The `EnhancedHarmonyAnalyzer` uses this information to provide smarter pairing advice:
- Solid + Solid: "Classic pairing"
- Solid + Pattern: "Check if solid matches a pattern color"
- Pattern + Pattern: "Ensure shared colors for coordination"

---

## Stage 3: Compatibility Scoring

This is the most sophisticated part of the system. How do you teach a computer what "looks good together"?

### The Training Data

The system can be trained on different fashion datasets:

**DeepFashion2** (Recommended - what this project uses):
- 191,961 high-resolution clothing images (468×624 pixels)
- Categories include: short/long sleeve tops, trousers, shorts, skirts, dresses
- Many images contain multiple items worn together (natural outfit pairs)
- Run `scripts/setup_deepfashion.py` to prepare this dataset

**Polyvore Outfits Dataset** (Alternative):
- 21,000+ expert-curated outfit combinations
- Items matched by fashion stylists
- Note: The original Polyvore website shut down, so images may be harder to obtain

Both datasets provide examples of items that go together, which the model uses to learn compatibility patterns.

### Siamese Networks and Triplet Learning

The compatibility model uses a **Siamese Network** with **triplet loss**. Here's the idea:

Imagine you're teaching someone what compatible clothes look like. You might say:
- "This blue shirt and these khaki pants go together" (positive pair)
- "This blue shirt does NOT go with these neon green shorts" (negative pair)

The network learns by looking at triplets:
1. An **anchor item** (e.g., a blue shirt)
2. A **positive match** (pants from the same outfit - known to be compatible)
3. A **negative match** (pants from a different outfit - likely not as compatible)

The goal: Learn to place compatible items **close together** in a mathematical space, and incompatible items **far apart**.

```
                    LEARNED EMBEDDING SPACE

         ○ Blue shirt           The blue shirt and khaki pants
         │                      end up close together because
         │  ○ Khaki pants       they were in the same outfit.
         │
         │
         │              ○ Neon green shorts
         │                      The neon shorts end up far away
                               because it wasn't paired with
                               the blue shirt in training data.
```

### Type-Aware Learning

The system is smart about categories. The "rules" for what makes a good top-pants combination are different from what makes good shoes-bag pairings. So it has separate learned parameters for each clothing type.

### The Final Score

For any two items, the system:
1. Looks up their learned representations (64 numbers each)
2. Calculates how similar they are (using cosine similarity)
3. Returns a score from 0 to 1 (higher = more compatible)

For a full outfit, it averages the compatibility scores of all pairs.

---

## Stage 4: Outfit Generation

Now comes the fun part - actually generating outfit recommendations.

### Outfit Templates

The system knows what a complete outfit looks like. Templates define valid structures:

- **Casual outfit**: top + bottom (+ optional shoes, accessories)
- **Complete outfit**: top + bottom + shoes + accessories
- **Dress outfit**: dress + shoes (+ optional accessories)

### Finding the Best Combinations

If you have 10 tops, 8 bottoms, 5 shoes, and 7 accessories, there are:
- 10 × 8 × 5 × 7 = **2,800 possible combinations**

The system could try all of them (exhaustive search), but for larger wardrobes, it uses **beam search** - a smart algorithm that explores the most promising combinations without checking every single one.

### Diversity

The recommendations should not all use the same favorite pair of jeans. The system adds **diversity weighting** to keep the list varied.

### Generating Explanations

Finally, the system explains *why* each outfit works:

```
OUTFIT RECOMMENDATION #1 (Score: 0.87)
────────────────────────────────────────
Items: Navy sweater + Khaki chinos + Brown loafers + Leather belt

Why it works:
  - Strong color harmony: The navy and brown create a classic
    complementary pairing
  - The neutral khaki bridges both pieces
  - All items share a "smart casual" aesthetic

Color palette: Navy (40%), Tan (35%), Brown (20%), White (5%)
```

---

## The Technology Stack

For those interested in the technical details:

| Component | Technology | Purpose |
|-----------|------------|---------|
| Deep Learning | PyTorch | The main AI framework |
| Image Recognition | EfficientNet-B0 | Pre-trained neural network for visual features |
| Image Processing | OpenCV, Pillow | Manipulating and analyzing images |
| Color Analysis | scikit-learn (K-means) | Clustering pixels to find dominant colors |
| Compatibility Model | Custom Siamese Network | Learning what items go together |

---

## The Complete Flow

Here's everything connected:

```
YOU TAKE A PHOTO OF YOUR SHIRT
            │
            ▼
    ┌───────────────┐
    │ Load Image    │ ← The photo is loaded and resized to 224×224 pixels
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Neural Net    │ ← EfficientNet analyzes the image
    │ (Recognition) │   Output: "This is a TOP" + visual features
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Color         │ ← K-means finds the main colors
    │ Extraction    │   Output: Navy blue, white trim
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Save to       │ ← Features are cached for fast access
    │ Wardrobe      │   (No need to re-analyze later)
    └───────────────┘
            │
    ════════════════════════════════════════════════
            │
    YOU ASK: "WHAT SHOULD I WEAR?"
            │
            ▼
    ┌───────────────┐
    │ Load Wardrobe │ ← All your items with their features
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Generate      │ ← Try combinations based on templates
    │ Combinations  │   (top + bottom + shoes + ...)
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Score Each    │ ← Siamese network computes compatibility
    │ Outfit        │   for each pair of items, then averages
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Rank & Select │ ← Sort by score, ensure variety
    │ Top 10        │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Explain       │ ← Generate human-readable explanations
    │ Recommendations│   "These work because..."
    └───────────────┘
            │
            ▼
    YOUR TOP 10 OUTFIT RECOMMENDATIONS!
```

---

## Key Concepts Glossary

**Neural Network**: A type of AI that learns patterns from examples, loosely inspired by how brains work.

**Embedding**: A way to represent something (like a clothing item) as a list of numbers that capture its important characteristics.

**K-means Clustering**: An algorithm that groups similar things together - used here to find dominant colors.

**Siamese Network**: A neural network architecture that learns to compare two things and determine how similar they are.

**Triplet Loss**: A training technique where the network learns from trios of examples: an anchor, a positive match, and a negative match.

**LAB Color Space**: A way of representing colors that matches human perception better than RGB.

**Cosine Similarity**: A mathematical way to measure how similar two lists of numbers are, returning a value from -1 to 1.

**Beam Search**: A smart search algorithm that explores promising options without trying every possibility.

---

## Why This Matters

This project connects several ML concepts I wanted to learn more deeply:

1. **Transfer Learning**: Using a model trained on millions of general images (ImageNet) and adapting it for fashion.

2. **Representation Learning**: Teaching computers to represent abstract concepts (like "style" or "compatibility") as numbers.

3. **Multi-modal Analysis**: Combining different types of information (visual appearance + color + category) for better results.

4. **Explainability**: Not just giving recommendations, but explaining *why* they scored well.

The same general ideas show up in recommendation systems for music, movies, shopping, and other ranking problems.

---

## Try It Yourself: The Jupyter Notebooks

The `/notebooks` folder contains 6 interactive notebooks that let you run each stage of the system yourself. Each notebook builds on the previous one, so run them in order.

---

### Notebook 1: Garment Recognition (`01_garment_recognition.ipynb`)

**Purpose:** Train and test the neural network that identifies what type of clothing is in a photo.

**What happens step by step:**

1. **Initialize the Classifier**
   - Loads a pre-trained EfficientNet-B0 model (already knows how to recognize general objects)
   - Adds a custom "head" that outputs 5 categories: tops, bottoms, dresses, shoes, accessories
   - The model has about 4 million learnable parameters

2. **Prepare the Dataset**
   - Creates a `FashionDataset` class that loads images from folders
   - Images are organized by category (e.g., `data/samples/tops/`, `data/samples/shoes/`)
   - **Data Augmentation**: Since we have limited training images, the notebook creates variations:
     - Random cropping (224×224 from 256×256)
     - Horizontal flipping
     - Color jittering (brightness, contrast, saturation)
   - This turns a small dataset into a larger, more diverse training set

3. **Training Loop**
   - Runs for 5 epochs (passes through all training data)
   - Uses **cross-entropy loss** to measure how wrong the predictions are
   - Uses **Adam optimizer** to adjust the model weights
   - Tracks training loss, validation loss, and accuracy
   - Includes **early stopping** - if validation loss stops improving, training ends

4. **Evaluation**
   - Tests the model on held-out images it hasn't seen before
   - Reports overall accuracy and per-class accuracy
   - Shows which categories the model is best/worst at recognizing

5. **Save the Model**
   - Saves the trained weights to `models/garment_classifier.pth`
   - This file is loaded by later notebooks

6. **Test on Individual Images**
   - Includes a function to predict on any image and show results
   - Displays the image with predicted category and confidence percentage

**What you'll learn:**
- How transfer learning works (starting from a pre-trained model)
- How image augmentation increases effective dataset size
- How training loops work with loss functions and optimizers
- How to evaluate classification accuracy

---

### Notebook 2: Attribute Extraction (`02_attribute_extraction.ipynb`)

**Purpose:** Extract color information from clothing images using computer vision techniques.

**What happens step by step:**

1. **Initialize the Color Extractor**
   - Creates a `ColorExtractor` configured to find 5 dominant colors
   - Uses LAB color space (perceptually uniform - see glossary)
   - Configured to remove backgrounds and filter out skin tones

2. **Test on a Sample Image**
   - Creates a simple test image (half navy blue, half white)
   - Extracts dominant colors and prints:
     - Color name (e.g., "blue", "white")
     - RGB values (e.g., [30, 50, 100])
     - LAB values (e.g., [25, 30, -60])
     - Percentage of the garment this color covers
     - Hex code for web display

3. **Visualize Extracted Colors**
   - Creates a horizontal bar chart showing each color
   - Bar width = percentage of garment
   - Bar color = actual color extracted
   - Labels show color name and percentage

4. **Color Harmony Analysis**
   - Tests the `ColorHarmonyAnalyzer` on different color pairs:
     - Blue + Orange: **Complementary** (opposite on color wheel)
     - Blue + Navy: **Analogous** (neighbors on color wheel)
     - Blue + Gray: **Neutral** (gray goes with everything)
   - Reports harmony type, score, and hue difference

5. **Create Color Feature Vectors**
   - Converts colors to a fixed-size vector (15 numbers)
   - Structure: 5 colors × 3 LAB values each = 15 dimensions
   - This vector can be fed into the neural network

6. **Combined Attribute Pipeline**
   - Loads the classifier from Notebook 1
   - Creates an `AttributePipeline` that combines classification + color extraction
   - For any image, outputs a `GarmentAttributes` object containing:
     - Item ID
     - Category and confidence
     - List of dominant colors with names
     - Visual embedding (1,280 numbers from the neural net)
     - Color vector (15 numbers)

7. **Test on Real Images**
   - If you have images in `data/samples/`, processes them
   - Shows each image alongside its extracted color palette

**What you'll learn:**
- How K-means clustering groups similar pixels
- Why LAB color space is better than RGB for perception
- How color theory (harmony types) is implemented in code
- How to combine multiple features into a single pipeline

---

### Notebook 3: Compatibility Modeling (`03_compatibility_modeling.ipynb`)

**Purpose:** Train the AI to understand which clothing items look good together.

**Prerequisites:** Run `python scripts/setup_deepfashion.py` first to prepare the dataset.

**What happens step by step:**

1. **Load the DeepFashion2 Dataset**
   - Loads outfit data from `data/deepfashion/`
   - Training set: ~8,500 outfits (real pairs + synthetic combinations)
   - Validation set: ~1,000 outfits
   - Items include tops, bottoms, and dresses with real images

2. **Extract Features for All Items**
   - For every item in the dataset:
     - Runs it through the classifier to get visual features (1,280 numbers)
     - Runs it through color extraction to get color features (15 numbers)
   - **Caching**: Features are saved to `data/processed/` so they do not have to be extracted every time

3. **Create Triplet Datasets**
   - Generates 50,000 training triplets, each containing:
     - **Anchor**: A clothing item (e.g., a blue shirt)
     - **Positive**: An item from the SAME outfit (compatible)
     - **Negative**: An item from a DIFFERENT outfit (not compatible)
   - Uses **hard negative mining**: picks negatives that are the same category as the positive (e.g., if positive is pants, negative is also pants from a different outfit)
   - This makes the task harder and forces the model to learn subtle compatibility cues

4. **Initialize the Siamese Network**
   - Creates a `SiameseCompatibilityNet` with:
     - Input: visual features (1,280) + color features (15) = 1,295 dimensions
     - Hidden layer: 512 neurons
     - Embedding layer: 256 neurons
     - **Type-aware projection**: Different 64-dimension outputs for each clothing category
     - L2 normalization: Forces embeddings onto a unit sphere
   - Total: ~1.2 million parameters

5. **Training with Triplet Loss**
   - For each triplet (anchor, positive, negative):
     - Compute embeddings for all three
     - Calculate distances: d(anchor, positive) and d(anchor, negative)
     - Loss = max(0, margin + d(anchor,pos) - d(anchor,neg))
   - The goal: Make d(anchor, positive) smaller than d(anchor, negative) by at least the margin (0.2)
   - Trains for up to 50 epochs with early stopping

6. **Plot Training History**
   - Shows loss curves (training and validation)
   - Shows triplet accuracy (how often positive is closer than negative)

7. **Evaluate with AUC-ROC**
   - Creates pairs of items labeled as compatible or not compatible
   - Model predicts compatibility scores
   - **AUC (Area Under Curve)**: Measures how well the model ranks compatible pairs above incompatible ones
   - AUC > 0.85 = Excellent, AUC > 0.75 = Good

8. **Save the Model**
   - Saves trained weights to `models/compatibility_model.pth`

9. **Test Compatibility Scoring**
   - Uses `CompatibilityScorer` to score sample pairs
   - Shows compatibility scores between different item combinations

**What you'll learn:**
- How Siamese networks learn to compare items
- How triplet loss teaches "this should be closer than that"
- Why hard negative mining improves learning
- How to evaluate ranking models with AUC

---

### Notebook 4: Outfit Generation (`04_outfit_generation.ipynb`)

**Purpose:** Use the trained models to generate outfit recommendations from a wardrobe.

**What happens step by step:**

1. **Load All Trained Models**
   - Loads the garment classifier from Notebook 1
   - Loads the compatibility model from Notebook 3
   - Creates the color extractor, attribute pipeline, and compatibility scorer

2. **Create a Sample Wardrobe**
   - Since you might not have real wardrobe images, creates synthetic items:
     - 5 tops: White T-shirt, Navy Shirt, Black Top, Gray Sweater, Light Blue Blouse
     - 4 bottoms: Dark Jeans, Khaki Pants, Black Pants, Gray Trousers
     - 3 shoes: White Sneakers, Black Shoes, Brown Boots
   - Each item is a colored rectangle with extracted features
   - Items are stored in a `Wardrobe` object organized by category

3. **Initialize the Outfit Generator**
   - Creates an `OutfitGenerator` with:
     - Minimum compatibility threshold: 0.3 (filter out bad combinations)
     - Diversity weight: 0.2 (encourage variety in recommendations)

4. **Define Outfit Templates**
   - Creates a "casual" template requiring: tops + bottoms + shoes
   - Templates ensure valid outfit structures

5. **Generate Outfit Recommendations**
   - The generator:
     - Enumerates all possible combinations (5 × 4 × 3 = 60 outfits)
     - Scores each outfit using the compatibility model
     - Ranks by score with diversity weighting
     - Returns top 10 recommendations

6. **Display Generated Outfits**
   - Shows each outfit as a row of colored blocks
   - Displays the outfit number and compatibility score

7. **Explain Recommendations**
   - Uses `OutfitExplainer` to analyze the top outfit:
     - Summary (what items, what score)
     - Color analysis (harmony type detected)
     - Compatibility breakdown (which pairs score highest)
     - Strengths (what makes this outfit work)
     - Suggestions (how it could be improved)
   - Uses `QuickExplainer` for one-line summaries of all outfits

8. **Build Outfit Around Specific Item**
   - Demonstrates "anchor" functionality:
     - Pick a specific item (e.g., the White T-shirt)
     - Generate outfits that MUST include that item
     - Useful when you've already decided on one piece

9. **Suggest Additions**
   - Given a partial outfit (just a top), suggests which bottoms pair best
   - Returns ranked list of suggestions with scores

**What you'll learn:**
- How outfit templates constrain valid combinations
- How compatibility scores are aggregated for full outfits
- How diversity weighting prevents repetitive recommendations
- How to generate explanations for AI decisions

---

### Notebook 5: Full Pipeline (`05_full_pipeline_demo.ipynb`)

**Purpose:** A full pipeline notebook that ties everything together with **multi-color classification** and shows how the system analyzes clothing patterns.

**What happens step by step:**

1. **Initialize the Complete System**
   - Loads all models (classifier, compatibility model)
   - Creates pipeline with **multi-color classification enabled**
   - Uses `EnhancedColorExtractor` and `EnhancedHarmonyAnalyzer`

2. **Multi-Color Classification Demo**
   - Creates sample images: solid, two-tone, and multi-color patterns
   - Shows how each is classified with pattern type and color summary
   - Demonstrates the `is_solid()`, `is_two_tone()`, `is_multicolor()` methods

3. **Enhanced Harmony Analysis**
   - Tests pattern combinations: solid+solid, solid+pattern, pattern+pattern
   - Shows harmony scores and pattern-specific advice

4. **Create Sample Wardrobe**
   - Builds a wardrobe with various color patterns
   - Each item shows its pattern classification (solid/two-tone/multi-color)
   - Visual display with color-coded borders by pattern type

5. **Pattern-Aware Outfit Generation**
   - Generates outfits with pattern information displayed
   - Shows pattern combination advice (e.g., "All solid - classic and safe!")

6. **Pattern-Based Filtering**
   - Filter outfits by pattern rules:
     - `all_solid`: Only solid-color outfits
     - `has_pattern`: Outfits with at least one patterned item
     - `one_pattern_max`: Maximum one patterned item per outfit

7. **Wardrobe Summary with Color Analysis**
   - Breakdown by category AND by color pattern
   - Most common colors in your wardrobe
   - Outfit combination potential

**What you'll learn:**
- How multi-color classification works
- How pattern combinations affect outfit scoring
- How to filter outfits by pattern rules
- How to interpret enhanced harmony analysis

---

### Notebook 6: Multi-Color Classification (`06_multicolor_classification.ipynb`)

**Purpose:** Deep dive into the multi-color classification system - understand and customize how the system categorizes color patterns.

**What happens step by step:**

1. **Define Color Patterns**
   - `ColorPattern` enum: SOLID, TWO_TONE, MULTI_COLOR
   - `ColorClassification` dataclass with primary, secondary, and accent colors
   - Configurable thresholds for classification

2. **MultiColorClassifier**
   - Thresholds:
     - `solid_threshold`: 0.85 (>= 85% = solid)
     - `two_tone_threshold`: 0.50 (50-85% with secondary = two-tone)
     - Below 50% or 3+ colors = multi-color
   - Customize these to match your preferences

3. **Test with Synthetic Images**
   - Create solid, two-tone, striped, and multi-color block images
   - Run classification and visualize results

4. **EnhancedColorExtractor**
   - Extends base `ColorExtractor` with classification
   - `extract_with_classification()` returns colors + classification
   - Quick methods: `is_solid()`, `is_multicolor()`, `is_two_tone()`

5. **EnhancedHarmonyAnalyzer**
   - Pattern-aware harmony scoring
   - Detects matching colors between patterned items
   - Penalizes pattern clashes, rewards color coordination

6. **Test on Real Images**
   - Process images from `data/samples/`
   - Visualize classification results

**What you'll learn:**
- How to customize classification thresholds
- How pattern detection affects harmony scoring
- How to integrate multi-color into your own applications
- The relationship between color distribution and pattern type

---

### Running the Notebooks

**Prerequisites:**
1. Install dependencies: `pip install -r requirements.txt`
2. Have Jupyter installed: `pip install jupyter`

**Order matters:**
- Run Notebook 1 first (trains the classifier)
- Run Notebook 3 before Notebook 4 (trains the compatibility model)
- Notebook 5 works best after all models are trained

**Start Jupyter:**
```bash
cd Fashion
jupyter notebook
```

Then open `notebooks/01_garment_recognition.ipynb` in your browser.

**Tips:**
- Each notebook takes 5-30 minutes to run fully
- You can skip training and use pre-trained models if available
- Add your own images to `data/samples/` organized by category

---

*This system was built using PyTorch and can be trained on the DeepFashion2 dataset (191,961 images) or the Polyvore Outfits dataset (21,889 outfit combinations).*
