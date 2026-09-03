"""FastAPI application for the Jimterpretability platform."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from jimterpretability.session import SessionRegistry
from api.routes import sessions, concepts, features, interventions


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = SessionRegistry()
    yield
    # Tear down all sessions cleanly on shutdown
    registry: SessionRegistry = app.state.registry
    for sid in list(registry._sessions.keys()):
        registry.remove(sid)


app = FastAPI(
    title="Jimterpretability API",
    version="0.1.0",
    description=(
        "Mechanistic interpretability platform: concept discovery, "
        "activation steering, output interception, and weight editing for open-weight LLMs."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api/v1")
app.include_router(concepts.router, prefix="/api/v1")
app.include_router(features.router, prefix="/api/v1")
app.include_router(interventions.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}
