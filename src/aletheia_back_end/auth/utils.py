from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from aletheia_back_end.app_settings import settings

basic_auth_scheme = HTTPBasic()


async def get_basic_auth_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(basic_auth_scheme)]
) -> str:
    """
    Validates Basic Authentication credentials against stored settings.
    Returns the username if successful, raises an HTTPException otherwise.

    This function is intended to be used as a FastAPI dependency to enforce
    Basic Authentication on protected endpoints. It receives credentials
    extracted by the HTTPBasic scheme and compares them to configured values.

    Args:
        credentials (HTTPBasicCredentials): The Basic Auth credentials
            (username and password) extracted from the request's
            'Authorization' header.

    Returns:
        str: The authenticated username if the credentials are valid.

    Raises:
        HTTPException: Raises an HTTPException with status code 401
            (UNAUTHORIZED) if the provided username or password do not match
            the expected values, and includes the 'WWW-Authenticate' header to
            prompt client login.
    """
    valid_username = credentials.username == settings.basic_auth_username
    valid_password = credentials.password == settings.basic_auth_password

    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username

# Dependency to use in routes
AuthUser = Annotated[str, Depends(get_basic_auth_user)]
