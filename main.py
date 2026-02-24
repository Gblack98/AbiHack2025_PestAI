import hashlib
import json
from contextlib import asynccontextmanager

import cloudinary
import google.api_core.exceptions
import google.generativeai as genai
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import (
    CACHE_TTL,
    CLOUDINARY_API_KEY_ENV,
    CLOUDINARY_API_SECRET,
    CLOUDINARY_CLOUD_NAME,
    GEMINI_API_KEYS,
    RATE_LIMIT,
)
from app.key_manager import KeyManager
from app.models import AIAnalysisResponse, AnalysisType
from app.prompts import DRONE_PROMPT, PLANT_PEST_PROMPT, SATELLITE_PROMPT
from app.services.cloudinary import crop_and_upload
from app.services.gemini import call_gemini

# --- Initialisation des services globaux ---
key_manager = KeyManager(GEMINI_API_KEYS)

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY_ENV,
    api_secret=CLOUDINARY_API_SECRET,
)

limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

PROMPTS = {
    AnalysisType.PLANT_PEST: PLANT_PEST_PROMPT,
    AnalysisType.SATELLITE_REMOTE_SENSING: SATELLITE_PROMPT,
    AnalysisType.DRONE_ANALYSIS: DRONE_PROMPT,
}

SUPPORTED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff"}
CROP_SUPPORTED_TYPES = {"image/jpeg", "image/jpg", "image/png"}


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix="pestai-cache")
    print("Cache initialisé. PestAI v12.0.0 prêt.")
    yield


# --- Application ---
app = FastAPI(
    title="PestAI - Unified Analysis Microservice",
    description=(
        "API v12.0 — Détection agronomique par IA (Plante, Satellite, Drone). "
        "Propulsé par Gemini 3 Flash Preview."
    ),
    version="12.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Cache key builder ---
def unified_key_builder(func, namespace: str = "", *, request: Request, response: Response, **kwargs):
    analysis_type_str = str(kwargs.get("analysis_type", "unknown"))
    file: UploadFile = kwargs["file"]
    content = file.file.read()
    file.file.seek(0)
    file_hash = hashlib.sha256(content).hexdigest()
    return f"{namespace}:{analysis_type_str}:{file_hash}"


# --- Endpoint principal ---
@app.post(
    "/api/v12/analyze",
    response_model=AIAnalysisResponse,
    summary="Analyse unifiée v12 — Plante / Satellite / Drone",
    tags=["PestAI v12"],
)
@limiter.limit(RATE_LIMIT)
@cache(namespace="pestai-v12", expire=CACHE_TTL, key_builder=unified_key_builder)
async def analyze(
    request: Request,
    response: Response,
    analysis_type: AnalysisType = Form(
        ...,
        description="Type d'analyse : PLANT_PEST, SATELLITE_REMOTE_SENSING ou DRONE_ANALYSIS.",
    ),
    file: UploadFile = File(
        ...,
        description="Image JPEG, JPG, PNG ou TIFF.",
    ),
):
    """
    Pipeline complet :
    1. Sélectionne le prompt expert selon le type d'analyse.
    2. Appelle Gemini 3 Flash avec rotation automatique de clés.
    3. Découpe les zones détectées et les uploade sur Cloudinary.
    4. Retourne le JSON validé par Pydantic.
    """
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Format non supporté. Formats acceptés : {', '.join(SUPPORTED_TYPES)}.",
        )

    image_bytes = await file.read()
    prompt = PROMPTS[analysis_type]
    image_part = {"mime_type": file.content_type, "data": image_bytes}
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

    try:
        raw_text = await call_gemini(prompt, image_part, generation_config, key_manager)
        analysis_data = json.loads(raw_text)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Le modèle IA a renvoyé une réponse non-JSON.",
        )
    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur API Google : {e.message}",
        )

    if analysis_data.get("detections") and file.content_type in CROP_SUPPORTED_TYPES:
        analysis_data["detections"] = crop_and_upload(
            image_bytes, analysis_data["detections"], folder="pestai_v12"
        )

    return analysis_data


# --- Health check ---
@app.get("/", include_in_schema=False)
def health():
    return {"status": "ok", "service": "PestAI v12.0.0", "model": "gemini-3-flash-preview"}
