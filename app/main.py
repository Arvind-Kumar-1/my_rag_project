from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.models import APIKey
from fastapi.openapi.utils import get_openapi
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

import time
import logging
from datetime import datetime

from .api import router
from .config import settings, logger

# FastAPI app instance
app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="""
    An LLM-Powered system to query documents in insurance, legal, HR, and compliance domains.

    ## Authentication
    All API endpoints (except health checks) require Bearer token authentication:

    ```
    Authorization: Bearer <api_key>
    ```

    You can also test via Swagger by clicking **Authorize** and entering your token.

    ## POST /hackrx/run
    - Requires: `documents` (URL) and `questions` (array)
    - Returns: array of answers
    """,
    version="2.0.0",
    contact={
        "name": "HackRx Team",
        "email": "team@hackrx.example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {"url": "https://your-domain.com", "description": "Production server"},
        {"url": "http://localhost:8000", "description": "Development server"}
    ]
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.is_development else ["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware for production
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com", "*.herokuapp.com", "*.railway.app"]
    )

# Auth Token Header (for Swagger)
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info(f"Response: {response.status_code} - Time: {duration:.2f}s")
        response.headers["X-Process-Time"] = str(duration)
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Request failed: {str(e)} - Time: {duration:.2f}s")
        raise

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.is_development else "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )

# Authorization middleware
# @app.middleware("http")
# async def enforce_auth_header(request: Request, call_next):
#     if request.url.path.startswith("/hackrx"):
#         auth_header = request.headers.get("Authorization")
#         if not auth_header or not auth_header.startswith("Bearer "):
#             return JSONResponse(
#                 status_code=HTTP_401_UNAUTHORIZED,
#                 content={
#                     "error": "Authorization header is required",
#                     "status_code": 401,
#                     "timestamp": datetime.now().isoformat(),
#                     "path": str(request.url.path)
#                 }
#             )
#         token = auth_header.split("Bearer ")[-1].strip()
#         if token != settings.AUTH_TOKEN:
#             return JSONResponse(
#                 status_code=HTTP_401_UNAUTHORIZED,
#                 content={
#                     "error": "Invalid API token",
#                     "status_code": 401,
#                     "timestamp": datetime.now().isoformat(),
#                     "path": str(request.url.path)
#                 }
#             )
#     return await call_next(request)

# Custom OpenAPI schema to enable token input in Swagger
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
            "description": "Enter your Bearer token like this: `Bearer <token>`"
        }
    }
    for path in openapi_schema["paths"].values():
        for op in path.values():
            op["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Router
app.include_router(router)

# Root endpoint
@app.get("/", tags=["Status"])
def read_root():
    return {
        "message": "Intelligent Query-Retrieval System API",
        "version": "2.0.0",
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "endpoints": {
            "main_endpoint": "/hackrx/run",
            "health": "/health",
            "models": "/models/info",
            "query_detailed": "/hackrx/run/detailed",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "authentication": {
            "type": "Bearer Token",
            "header": "Authorization: Bearer <api_key>",
            "required_for": "All endpoints except /health and /"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status", tags=["Status"])
def get_status():
    return {
        "api_status": "healthy",
        "environment": settings.ENVIRONMENT,
        "debug_mode": settings.DEBUG,
        "models": {
            "embedding": settings.EMBEDDING_MODEL,
            "llm": settings.LLM_MODEL
        },
        "configuration": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "max_chunks": settings.MAX_CHUNKS_PER_QUERY,
            "timeout": settings.REQUEST_TIMEOUT
        },
        "timestamp": datetime.now().isoformat()
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Intelligent Query-Retrieval System...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Auth token configured: {'Yes' if settings.AUTH_TOKEN != 'default-token-for-development' else 'Default (update for production)'}")
    if settings.is_production:
        try:
            from .processor import RAGProcessor
            processor = RAGProcessor()
            logger.info("Models pre-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load models: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Intelligent Query-Retrieval System...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
