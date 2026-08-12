# Fashion Web Demo

A local, mobile-friendly web app for the outfit recommendation pipeline. Take
photos of closet items, confirm the predicted category, then ask for outfit
ideas.

The ML code lives in `../src/`. This folder adds a small API and mobile-friendly UI on top of it.

## What it does

| Screen | What happens |
|---|---|
| **Login** | Single password. Session cookie, valid 30 days. |
| **Closet** | Grid of every item you've added, filterable by category. Tap an item to delete. |
| **Add (FAB +)** | Camera or photo-library picker -> backend runs `AttributePipeline.process()` -> returns predicted category + color summary -> you confirm or override -> saved. |
| **Today** | Three high-scoring outfits, each with a one-line explanation from `OutfitExplainer`. Tap "Wear this" to log it so future suggestions avoid repeats. "Build around..." lets you pin one item. |

## One-time setup

1. Copy the env file and pick a password:
   ```bash
   cd webapp
   cp .env.example .env
   $EDITOR .env  # set FASHION_PASSWORD and FASHION_SECRET_KEY
   ```

2. Make sure the trained model weights are at `../models/garment_classifier.pth`
   and `../models/compatibility_model.pth`.

## Running

```bash
cd webapp
docker compose up --build
```

First build takes ~5 minutes (downloads CPU PyTorch wheels). Subsequent starts
take ~10 seconds.

- Web UI: <http://localhost:8080>
- API (for curl/debug): <http://localhost:8000/api/health>

## Using it from your iPhone

The phone and laptop need to be on the same Wi-Fi.

1. On the laptop, find its LAN IP:
   ```bash
   ipconfig getifaddr en0    # macOS Wi-Fi
   ```
2. On the phone, open Safari and visit `http://<that-ip>:8080`.
3. Tap the Share button -> **Add to Home Screen**. You'll get an app icon
   that opens fullscreen, no Safari chrome.
4. The "Take photo" button opens the rear camera directly.

If "Take photo" opens the file picker instead of the camera, choose from the
photo library. Some iOS versions are picky about camera access on non-HTTPS
local sites.

## Data layout

Everything the app writes lives in `webapp/data/`:

```
webapp/data/
├── wardrobe/
│   ├── images/         # original photos
│   ├── cache/          # pickled visual+color features per item
│   └── metadata.json   # category + user-provided overrides
└── wear_log.json       # append-only list of outfits you've worn
```

Back this folder up if you want to keep a closet between runs. Delete it to reset the app.

## Architecture notes

- `backend/app/ml_service.py` loads the ML modules once at FastAPI startup.
  Inference runs behind a lock because the app is meant for one local user.
- The existing `WardrobeManager` from `src/data/wardrobe.py` already handles
  per-item persistence, so the API layer is thin.
- `OutfitExplainer.summary` and `color_analysis` strings are what get shown on
  the Today card.
- Recency-aware ranking: the last 7 days of `wear_log.json` lightly penalize
  outfits with items that were just worn, so the same combination does not show
  up every time.

## Development

If you want hot-reload while iterating on `src/` or `backend/app/`:

```bash
# in webapp/
docker compose up --build
# then edit files in src/ or webapp/backend/app/; restart the backend container
# (volumes mount the source live; uvicorn does not auto-reload in this setup
# unless you add --reload).
```

For the frontend, the build is baked into the image. For frontend iteration
without Docker, use Node 20+:

```bash
cd webapp/frontend
npm install
VITE_API_TARGET=http://localhost:8000 npm run dev
# Visit http://localhost:5173
# Backend must be running (docker compose up backend).
```
