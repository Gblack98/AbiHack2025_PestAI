import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

import cloudinary
import google.api_core.exceptions
import google.generativeai as genai
import httpx
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
    GEMINI_MODEL,
    GEMINI_VOICE_KEYS,
    RATE_LIMIT,
)
from app.key_manager import KeyManager
from app.models import AIAnalysisResponse, AnalysisType, VoiceAudioRequest, VoiceTextResponse
from app.prompts import DRONE_PROMPT, PLANT_PEST_PROMPT, SATELLITE_PROMPT, WOLOF_SUMMARY_PROMPT
from app.services.cloudinary import crop_and_upload
from app.services.gemini import call_gemini

# --- Initialisation des services globaux ---
key_manager = KeyManager(GEMINI_API_KEYS)
voice_key_manager = KeyManager(GEMINI_VOICE_KEYS)

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

# Chaîne de fallback pour text-only Wolof (chaque modèle a son propre quota)
GEMINI_TEXT_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1/models"


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
    3. Découpe les zones détectées et les uploade sur Cloudinary (parallèle).
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
        loop = asyncio.get_event_loop()
        analysis_data["detections"] = await loop.run_in_executor(
            None, crop_and_upload, image_bytes, analysis_data["detections"], "pestai_v12"
        )

    return analysis_data


# --- Endpoint texte Wolof (synthèse audio gérée côté mobile via HF Space xTTS) ---
@app.post(
    "/api/v12/voice-text",
    response_model=VoiceTextResponse,
    summary="Génère un résumé vocal en Wolof (texte seul)",
    tags=["PestAI v12"],
)
@limiter.limit(RATE_LIMIT)
async def voice_text(request: Request, body: VoiceAudioRequest):
    """
    Génère un court résumé en Wolof à partir de l'analyse agronomique.
    La synthèse audio est réalisée côté mobile via le Space HF xTTS GalsenAI (gratuit).
    Utilise gemini-2.0-flash via l'API REST directe (plus fiable pour text-only).
    """
    analysis_str = body.json(ensure_ascii=False)
    prompt = WOLOF_SUMMARY_PROMPT.format(analysis=analysis_str)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 300},
    }

    # Essaie chaque modèle dans l'ordre, avec rotation de clés sur 429.
    # Chaque modèle a son propre pool de quota journalier.
    last_error: str = "Tous les modèles Gemini sont épuisés."
    async with httpx.AsyncClient(timeout=25.0) as client:
        for model in GEMINI_TEXT_MODELS:
            url = f"{GEMINI_REST_BASE}/{model}:generateContent"
            num_keys = len(voice_key_manager.keys)
            for _ in range(num_keys):
                api_key = voice_key_manager.get_current_key()
                try:
                    r = await client.post(url, params={"key": api_key}, json=payload)
                except httpx.RequestError as e:
                    raise HTTPException(status_code=502, detail=f"Connexion Gemini impossible : {e}")

                if r.status_code == 429:
                    await voice_key_manager.rotate()
                    last_error = f"Quota 429 [{model}] clé ...{api_key[-4:]}"
                    continue

                try:
                    r.raise_for_status()
                    data = r.json()
                    wolof_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return VoiceTextResponse(text=wolof_text)
                except httpx.HTTPStatusError as e:
                    raise HTTPException(status_code=503, detail=f"Erreur API Gemini : {e.response.text}")
                except (KeyError, IndexError):
                    raise HTTPException(status_code=502, detail="Réponse Gemini invalide.")

    raise HTTPException(status_code=503, detail=last_error)


# --- Health check ---
@app.get("/", include_in_schema=False)
def health():
    return {"status": "ok", "service": "PestAI v12.0.0", "model": "gemini-3-flash-preview"}
