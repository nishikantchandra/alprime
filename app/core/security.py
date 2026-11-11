"""Optional API key authentication dependency controlled via `AUTH_ENABLED`."""

import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.api.dependencies import get_app_settings
from app.config import Settings

# Declared as a security scheme so Swagger UI shows an "Authorize" button for the key.
# auto_error=False lets the dependency below decide, because auth can be switched off.
_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Required only when the server runs with AUTH_ENABLED=true.",
)


def verify_api_key(
    x_api_key: str | None = Security(_api_key_header),
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
