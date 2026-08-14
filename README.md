# Fashion Outfit Recommendation

This is a machine-learning project I built to recognize clothing items, extract their colors, and recommend outfits from a small personal wardrobe. I started with notebooks while I was building the models, then added a phone-friendly web app so the project is easier to try.

## What It Does

- Classifies garments into tops, bottoms, dresses, shoes, and accessories.
- Extracts dominant colors from each item using LAB color space and K-means.
- Detects solid, two-tone, and multi-color items so busy patterns are handled more carefully.
- Scores outfit compatibility with a Siamese neural network.
- Generates outfit recommendations with short explanations.
- Includes a local FastAPI + React web app for trying the system from a laptop or phone.

## Ways to Run It

There are two main ways to use the project:

1. **Notebooks:** start with `notebooks/05_full_pipeline_demo.ipynb`.
2. **Web app:** run the mobile closet app in `webapp/`.

The web app is the easiest version to try live because it lets you add clothing photos, check the model's category guess, and generate outfit suggestions.

## Project Layout

```text
Fashion/
├── src/                    # Core ML pipeline
│   ├── attributes/         # Color extraction and pattern detection
│   ├── compatibility/      # Compatibility model and scoring
│   ├── data/               # Wardrobe storage helpers
│   ├── generation/         # Outfit generation and explanations
│   └── recognition/        # Garment classifier
├── notebooks/              # Training and walkthrough notebooks
├── webapp/                 # FastAPI backend + React frontend
├── models/                 # Trained model weights
├── configs/                # Training and pipeline configuration
├── scripts/                # Dataset setup scripts
└── docs/                   # Longer explanation of the approach
```

## Quick Start

For the notebook version:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open `notebooks/05_full_pipeline_demo.ipynb`.

For the web app:

```bash
cd webapp
cp .env.example .env
# edit .env and set FASHION_PASSWORD and FASHION_SECRET_KEY
docker compose up --build
```

Open `http://localhost:8080`.

## Data and Models

The repository keeps source code, notebooks, docs, sample images, and the trained model weights. Large datasets and generated caches are ignored:

- `data/polyvore/`
- `data/deepfashion/`
- `data/deepfashion2/`
- `data/fashion_mnist/`
- `data/processed/`
- `webapp/data/`

If those folders exist locally, they are from training or from trying the app. They can be regenerated and would make the repo too large.

## Design Notes

I kept the project organized around the ML pipeline: recognition, color extraction, compatibility scoring, and outfit generation. The notebooks show the experiments, while the web app uses the same code in a more practical interface.
