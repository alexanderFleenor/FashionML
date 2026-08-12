"""FastAPI entry point for the Fashion web app."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .config import settings
from .ml_service import MLService, ml_service_holder
from .routes import auth, items, outfits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fashion.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading ML models from %s ...", settings.MODELS_DIR)
    ml_service_holder.instance = MLService.load(
        models_dir=settings.MODELS_DIR,
        wardrobe_dir=settings.WARDROBE_DIR,
    )
    log.info("ML service ready. %d items in wardrobe.", len(ml_service_holder.instance.manager.wardrobe))
    yield
    log.info("Shutting down.")


app = FastAPI(title="Fashion Demo", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY,
    session_cookie=settings.SESSION_COOKIE_NAME,
    max_age=settings.SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=False,  # local app runs over http
)

if settings.DEV_MODE:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(items.router, prefix="/api/items", tags=["items"])
app.include_router(outfits.router, prefix="/api/outfits", tags=["outfits"])


@app.get("/api/health")
def health():
    return {"ok": True}
