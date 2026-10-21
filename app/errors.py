"""Centralized error handling with consistent JSON responses."""

import uuid
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger("docmind")


class AppError(Exception):
    """Base application error."""
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code


def _error_response(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def register_error_handlers(app):
    """Register all error handlers on the FastAPI app."""

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"[{request_id}] AppError: {exc.code} - {exc.message}")
        return _error_response(exc.status_code, exc.code, exc.message, request_id)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"[{request_id}] HTTPException: {exc.status_code} - {exc.detail}")
        return _error_response(
            exc.status_code,
            f"HTTP_{exc.status_code}",
            str(exc.detail),
            request_id,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "unknown")
        errors = exc.errors()
        # Summarize validation errors without exposing internals
        messages = []
        for err in errors:
            loc = " -> ".join(str(l) for l in err.get("loc", []))
            messages.append(f"{loc}: {err.get('msg', 'invalid')}")
        message = "; ".join(messages)
        logger.warning(f"[{request_id}] Validation error: {message}")
        return _error_response(422, "VALIDATION_ERROR", message, request_id)

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(f"[{request_id}] Unhandled exception: {type(exc).__name__}")
        return _error_response(
            500,
            "INTERNAL_ERROR",
            "An unexpected error occurred. Please try again.",
            request_id,
        )
