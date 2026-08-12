# Quick Start

Get the Fashion Outfit Recommendation system running in minutes.

## What's Included

This project comes with small trained model files so you can use it immediately without training:
- `models/garment_classifier.pth` - Identifies clothing types (tops, bottoms, shoes, etc.)
- `models/compatibility_model.pth` - Scores how well items go together

## Windows Setup

1. **Install Python 3.9+** from https://python.org
   - **Important:** Check "Add Python to PATH" during installation

2. **Double-click `setup_windows.bat`**
   - This creates a virtual environment and installs all dependencies
   - Takes about 5-10 minutes

3. **Run Jupyter:**
   ```
   venv\Scripts\activate
   jupyter notebook
   ```

4. **Open a notebook:**
   - `notebooks/05_full_pipeline_demo.ipynb` - Full pipeline with outfit recommendations
   - `notebooks/06_multicolor_classification.ipynb` - Explore the color pattern detection

## Mac/Linux Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter
jupyter notebook
```

## Which Notebooks to Run

| Notebook | Purpose | Need to Run? |
|----------|---------|--------------|
| 01 | Train garment classifier | **No** - already trained |
| 02 | Color extraction walkthrough | Optional |
| 03 | Train compatibility model | **No** - already trained |
| 04 | Outfit generation walkthrough | Optional |
| **05** | **Full pipeline** | **Start here!** |
| 06 | Multi-color classification deep-dive | Optional |

## Features

- **Garment classification** - identifies tops, bottoms, dresses, shoes, and accessories
- **Color extraction** - finds dominant colors in clothing images
- **Multi-color detection** - classifies items as solid, two-tone, or multi-color
- **Outfit recommendations** - suggests compatible outfit combinations
- **Pattern-aware scoring** - handles patterned pieces more carefully than solid pieces

## Troubleshooting

**"torch not found" or similar errors:**
- Make sure the virtual environment is activated
- On Windows: `venv\Scripts\activate`
- On Mac/Linux: `source venv/bin/activate`

**Jupyter won't start:**
```
pip install jupyter
```

**Models not loading:**
- Make sure the `models/` folder contains:
  - `garment_classifier.pth`
  - `compatibility_model.pth`

## Project Structure

```
Fashion/
├── models/                 # Trained model weights
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── attributes/         # Color extraction & multicolor
│   ├── compatibility/      # Outfit scoring
│   ├── generation/         # Outfit recommendations
│   └── recognition/        # Garment classification
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── setup_windows.bat       # Windows setup script
└── QUICKSTART.md          # This file
```

## Before Uploading to GitHub

Large local datasets and generated caches should stay out of the repository. The root `.gitignore` already excludes `data/polyvore/`, `data/deepfashion*/`, `data/fashion_mnist/`, `data/processed/`, `webapp/data/`, Python bytecode, virtual environments, and local `.env` files.
