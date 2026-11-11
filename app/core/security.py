"""Optional API key authentication dependency controlled via `AUTH_ENABLED`."""

import secrets

from fastapi import Depends, Header, HTTPException, status

from app.api.dependencies import get_app_settings
from app.config import Settings


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_app_settings),
) -> None:
    """Enforce `X-API-Key` validation when `AUTH_ENABLED=true`."""
    if not settings.auth_enabled:
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
