"""
PHASE 5 TIER-0: Security Utilities
Bearer token authentication and rate limiting
"""

from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Bearer token security scheme
security_scheme = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme),
    required_token: str = None,
    enabled: bool = True
) -> bool:
    """
    Verify Bearer token
    
    Args:
        credentials: HTTP authorization credentials
        required_token: Expected token value
        enabled: Whether authentication is enabled
    
    Returns:
        True if token is valid or auth is disabled
    
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not enabled:
        return True
    
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if credentials.credentials != required_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request
    
    Args:
        request: FastAPI request object
    
    Returns:
        Client IP address
    """
    # Check for X-Forwarded-For header (proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP if multiple
        return forwarded_for.split(",")[0].strip()
    
    # Check for X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client
    return request.client.host if request.client else "unknown"

