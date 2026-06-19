from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes.skills import router as skills_router
from backend.routes.learning_path import router as learning_path_router
from backend.skill_resources.main import router as resources_router


app = FastAPI(title="SkillSynapse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Starlette's ServerErrorMiddleware bypasses CORSMiddleware when it sends a 500,
    # so the browser sees no Access-Control-Allow-Origin header and reports a misleading
    # CORS error. Registering here keeps the response inside FastAPI's stack (inside
    # CORSMiddleware), so CORS headers are always present.
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


@app.get("/health")
@app.get("/healthz")
async def health():
    return {"status": "ok"}


app.include_router(skills_router)
app.include_router(learning_path_router)
app.include_router(resources_router)
