import os
import json
import hashlib
import io
import itertools
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.api_core.exceptions

import cloudinary
import cloudinary.uploader
from PIL import Image

load_dotenv()

# --- KeyManager ---
class KeyManager:
    def __init__(self, keys: List[str]):
        if not keys or all(k == '' for k in keys):
            raise ValueError("La liste des clés API Gemini ne peut pas être vide.")
        self.keys = keys
        self._key_iterator = itertools.cycle(keys)
        self._current_key = next(self._key_iterator)
        print(f"Gestionnaire de clés initialisé avec {len(self.keys)} clé(s).")

    def get_current_key(self) -> str:
        return self._current_key

    def switch_to_next_key(self) -> str:
        self._current_key = next(self._key_iterator)
        print(f"Limite de quota atteinte. Passage à la clé API suivante : ...{self._current_key[-4:]}")
        return self._current_key

gemini_api_keys_str = os.getenv("GEMINI_API_KEYS")
if not gemini_api_keys_str:
    raise ValueError("Variable d'environnement GEMINI_API_KEYS non trouvée.")
gemini_api_keys = [key.strip() for key in gemini_api_keys_str.split(',') if key.strip()]
key_manager = KeyManager(gemini_api_keys)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache en mémoire initialisé. Le service v8.5 est prêt à analyser.")
    yield


app = FastAPI(
    title="PestAI - IA Analysis Microservice",
    description="API v8.5. Version de référence — prompt universel, rotation de clés Gemini.",
    version="8.5.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Modèles Pydantic ---
class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class BoundingBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)

class SolutionDetail(BaseModel):
    solution: str
    details: str
    source: Optional[str] = None

class RecommendationsGroup(BaseModel):
    biological: List[SolutionDetail]
    chemical: List[SolutionDetail]
    cultural: List[SolutionDetail]

class DetailedInfo(BaseModel):
    description: str
    impact: str
    recommendations: RecommendationsGroup
    knowledgeBaseTags: List[str]

class Detection(BaseModel):
    className: str
    confidenceScore: float
    severity: SeverityLevel
    boundingBox: BoundingBox
    details: DetailedInfo
    croppedImageUrl: Optional[str] = Field(None)

class AnalysisSubject(BaseModel):
    subjectType: str
    description: str
    confidence: float

class AIAnalysisResponse(BaseModel):
    subject: AnalysisSubject
    detections: List[Detection]


# --- Prompt universel ---
UNIVERSAL_PROMPT = """
Tu es 'PestAI-Core', un moteur d'analyse d'images agronomiques de classe mondiale.
Ta seule fonction est de recevoir une image et de retourner une analyse experte complète.

Identifie le sujet ('PLANT', 'PEST', ou 'UNKNOWN'), chaque problème détecté,
sa sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'), et fournis des recommandations groupées.

Réponds EXCLUSIVEMENT en JSON avec ce schéma :
{
  "subject": { "subjectType": "string", "description": "string", "confidence": "float" },
  "detections": [
    {
      "className": "string",
      "confidenceScore": "float",
      "severity": "string",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string",
        "impact": "string",
        "recommendations": {
          "biological": [ { "solution": "string", "details": "string", "source": "string|null" } ],
          "chemical":   [ { "solution": "string", "details": "string", "source": "string|null" } ],
          "cultural":   [ { "solution": "string", "details": "string", "source": "string|null" } ]
        },
        "knowledgeBaseTags": ["string"]
      }
    }
  ]
}
"""


async def generate_gemini_analysis_with_key_rotation(image_part: dict, config: genai.types.GenerationConfig):
    initial_key = key_manager.get_current_key()
    for _ in range(len(key_manager.keys)):
        try:
            current_key = key_manager.get_current_key()
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            response = await model.generate_content_async(
                [UNIVERSAL_PROMPT, image_part],
                generation_config=config,
                request_options={'timeout': 120}
            )
            return response
        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.PermissionDenied) as e:
            print(f"Erreur quota pour la clé ...{current_key[-4:]} : {e}")
            key_manager.switch_to_next_key()
            if key_manager.get_current_key() == initial_key:
                raise HTTPException(status_code=429, detail="Toutes les clés API Gemini ont dépassé leur quota.")
    raise HTTPException(status_code=503, detail="Échec de l'analyse IA après rotation de toutes les clés.")


def image_key_builder(func, namespace: str = "", *, request: Request, response: Response, **kwargs):
    file_content = kwargs["file"].file.read()
    kwargs["file"].file.seek(0)
    file_hash = hashlib.sha256(file_content).hexdigest()
    return f"{namespace}:{file_hash}"


@app.post(
    "/api/v8/analyze-image",
    response_model=AIAnalysisResponse,
    summary="Analyse IA universelle v8.5 (version de référence)",
    tags=["PestAI v8.5 (référence)"],
)
@limiter.limit("15/minute")
@cache(namespace="pestai-v8", expire=86400, key_builder=image_key_builder)
async def analyze_image_endpoint(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="Fichier image (JPEG, PNG)."),
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Format non supporté. Utilisez JPEG ou PNG.")

    image_bytes = await file.read()

    try:
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        gemini_response = await generate_gemini_analysis_with_key_rotation(image_part, generation_config)
        analysis_data = json.loads(gemini_response.text)

        if analysis_data.get("detections"):
            original_image = Image.open(io.BytesIO(image_bytes))
            width, height = original_image.size
            for detection in analysis_data["detections"]:
                detection.setdefault("croppedImageUrl", None)
                bbox = detection.get("boundingBox")
                if not bbox:
                    continue
                try:
                    coords = (
                        int(bbox["x_min"] * width), int(bbox["y_min"] * height),
                        int(bbox["x_max"] * width), int(bbox["y_max"] * height),
                    )
                    if coords[0] >= coords[2] or coords[1] >= coords[3]:
                        continue
                    cropped = original_image.crop(coords)
                    buffer = io.BytesIO()
                    cropped.save(buffer, format="PNG")
                    buffer.seek(0)
                    result = cloudinary.uploader.upload(buffer, folder="pestai_detections")
                    detection["croppedImageUrl"] = result.get("secure_url")
                except Exception as e:
                    print(f"Avertissement crop '{detection.get('className', '?')}' : {e}")

        return analysis_data

    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Réponse non-JSON reçue du modèle IA.")
    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=503, detail=f"Erreur API Google : {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.get("/", include_in_schema=False)
def read_root():
    return {"status": "ok", "service": "PestAI v8.5 (référence)", "model": "gemini-3-flash-preview"}
