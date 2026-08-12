"""Runtime configuration pulled from environment variables."""

import os
from pathlib import Path


class Settings:
    # Where the ML pipeline persists images + features + metadata.json
    DATA_DIR: Path = Path(os.environ.get("FASHION_DATA_DIR", "/app-data")).resolve()
    WARDROBE_DIR: Path = DATA_DIR / "wardrobe"
    WEAR_LOG_PATH: Path = DATA_DIR / "wear_log.json"

    # Where the trained .pth files live
    MODELS_DIR: Path = Path(os.environ.get("FASHION_MODELS_DIR", "/app/models")).resolve()

    # Local auth: one password, optionally pre-hashed with bcrypt.
    # If FASHION_PASSWORD is set, it's the plaintext password to compare against.
    # If FASHION_PASSWORD_HASH is set, it's a bcrypt hash.
    PASSWORD: str | None = os.environ.get("FASHION_PASSWORD")
    PASSWORD_HASH: str | None = os.environ.get("FASHION_PASSWORD_HASH")

    # Used to sign session cookies. Must be stable across restarts.
    SECRET_KEY: str = os.environ.get("FASHION_SECRET_KEY", "change-me-for-local-use")

    SESSION_COOKIE_NAME: str = "fashion_session"
    SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 24 * 30  # 30 days

    # CORS / dev convenience
    DEV_MODE: bool = os.environ.get("FASHION_DEV", "0") == "1"


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.WARDROBE_DIR.mkdir(parents=True, exist_ok=True)
