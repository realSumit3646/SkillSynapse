from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.skills import router as skills_router
from backend.routes.learning_path import router as learning_path_router


app = FastAPI(title="SkillSynapse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(skills_router)
app.include_router(learning_path_router)
