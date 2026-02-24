import os
import json
import hashlib
import asyncio
import io
import itertools
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

# --- Dépendances ---
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, Form
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Dépendances pour l'optimisation des quotas ---
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import google.api_core.exceptions

# --- Dépendances pour le traitement d'image et Cloudinary ---
import cloudinary
import cloudinary.uploader
from PIL import Image

# --- Configuration Initiale ---
load_dotenv()

# --- GESTIONNAIRE DE CLÉS API GEMINI ---
class KeyManager:
    """
    Gestionnaire de clés API asynchrone et "coroutine-safe"
    pour la rotation des clés en cas d'erreur de quota.
    """
    def __init__(self, keys: List[str]):
        if not keys or all(k == '' for k in keys):
            raise ValueError("La liste des clés API Gemini ne peut pas être vide.")
        self.keys = keys
        self._key_iterator = itertools.cycle(keys)
        self._current_key = next(self._key_iterator)
        self._lock = asyncio.Lock()
        print(f"Gestionnaire de clés initialisé avec {len(self.keys)} clé(s).")

    def get_current_key(self) -> str:
        return self._current_key

    async def switch_to_next_key_async(self) -> str:
        async with self._lock:
            key_before_switch = self._current_key
            self._current_key = next(self._key_iterator)
            if key_before_switch != self._current_key:
                print(f"Limite de quota atteinte. Passage à la clé API suivante : ...{self._current_key[-4:]}")
            return self._current_key

# --- Configuration des clés Gemini ---
gemini_api_keys_str = os.getenv("GEMINI_API_KEYS")
if not gemini_api_keys_str:
    raise ValueError("Variable d'environnement GEMINI_API_KEYS non trouvée.")
gemini_api_keys = [key.strip() for key in gemini_api_keys_str.split(',') if key.strip()]
key_manager = KeyManager(gemini_api_keys)

# --- Configuration Cloudinary ---
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# --- Configuration du Rate Limiter ---
limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])


# --- Lifespan (remplace @app.on_event("startup")) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache en mémoire initialisé. Service unifié v11.0.0 prêt.")
    yield


# --- Initialisation de l'Application FastAPI ---
app = FastAPI(
    title="PestAI - Unified Analysis Microservice",
    description="API v11.0. Service unifié (Plante, Satellite, Drone)",
    version="11.0.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Types d'analyse possibles ---
class AnalysisType(str, Enum):
    PLANT_PEST = "PLANT_PEST"
    SATELLITE_REMOTE_SENSING = "SATELLITE_REMOTE_SENSING"
    DRONE_ANALYSIS = "DRONE_ANALYSIS"

# --- Structures de Données Pydantic ---
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

# --- Banque de Prompts v11 ---

PLANT_PEST_PROMPT = """
Tu es 'PestAI-Core', un moteur d'analyse d'images agronomiques de classe mondiale. Ta fonction est l'analyse d'images de PROXIMITÉ (feuilles, tiges, insectes).

**TA MISSION :**
1.  **Identifier le Sujet Principal :** 'PLANT', 'PEST', ou 'UNKNOWN'. Si l'image n'est pas claire ou pas agricole, utilise 'UNKNOWN'.
2.  **Mener une Analyse Complète :**
    - Identifie l'espèce du sujet et chaque problème (maladie/ravageur).
    - Pour chaque détection, tu DOIS évaluer sa sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    - Tu DOIS générer des `knowledgeBaseTags` pertinents.
3.  **Fournir une Réponse JSON Strictement Structurée :** Ta réponse doit être EXCLUSIVEMENT au format JSON. Ne renvoie AUCUN texte avant ou après.

{
  "subject": {
    "subjectType": "string ('PLANT', 'PEST', or 'UNKNOWN')",
    "description": "string (ex: 'Plant de Maïs (Zea mays)')",
    "confidence": "float (0.0-1.0)"
  },
  "detections": [
    {
      "className": "string (Nom du problème, ex: 'Rouille commune du maïs')",
      "confidenceScore": "float",
      "severity": "string ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string (Description simple du problème)",
        "impact": "string (Impact simple sur la plante)",
        "recommendations": {
          "biological": [ { "solution": "string", "details": "string", "source": "string (URL)" } ],
          "chemical": [ { "solution": "string", "details": "string", "source": "string (URL)" } ],
          "cultural": [ { "solution": "string", "details": "string", "source": "string (URL)" } ]
        },
        "knowledgeBaseTags": [ "string" ]
      }
    }
  ]
}

**RÈGLES D'OR v11 :**
- **LANGAGE (TRÈS IMPORTANT) :** Les champs `description`, `impact`, `solution` et `details` seront traduits en Wolof et lus par une IA vocale (TTS). Utilise des phrases **simples, courtes et directes**. Évite le jargon complexe.
- **CONCISION :** Les descriptions et impacts doivent faire 1-2 phrases maximum.
- **SÉVÉRITÉ :** Le champ `severity` est obligatoire.
- **BOUNDING BOX :** Les boîtes doivent cibler la lésion ou le ravageur.
"""

SATELLITE_PROMPT = """
Tu es 'PestAI-RemoteSensing', un moteur d'analyse expert en agronomie et en télédétection. Ta fonction est l'analyse d'images SATELLITES (ex: Sentinel-2, Landsat) de parcelles agricoles.

**TA MISSION :**
1.  **Identifier le Sujet Principal :** Toujours 'SATELLITE_PLOT'.
2.  **Mener une Analyse Diagnostique Zonale :**
    - Identifie les problèmes majeurs (ex: 'Stress Hydrique', 'Faible Vigueur (NDVI)', 'Déficience Azotée', 'Salinisation').
    - Évalue la sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') pour chaque problème.
    - Délimite la zone affectée avec une `boundingBox`.
    - Génère des `knowledgeBaseTags`.
3.  **Fournir une Réponse JSON Strictement Structurée :** EXCLUSIVEMENT au format JSON.

{
  "subject": {
    "subjectType": "string (Toujours 'SATELLITE_PLOT')",
    "description": "string (ex: 'Analyse de parcelle agricole.')",
    "confidence": 1.0
  },
  "detections": [
    {
      "className": "string (Nom du diagnostic, ex: 'Stress Hydrique')",
      "confidenceScore": "float",
      "severity": "string ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string (Description simple du diagnostic)",
        "impact": "string (Impact simple sur le rendement)",
        "recommendations": {
          "biological": [],
          "chemical": [ { "solution": "string", "details": "string", "source": null } ],
          "cultural": [ { "solution": "string", "details": "string", "source": null } ]
        },
        "knowledgeBaseTags": [ "string" ]
      }
    }
  ]
}

**RÈGLES D'OR v11 :**
- **LANGAGE (TRÈS IMPORTANT) :** Utilise des phrases simples pour le TTS Wolof.
- **CONCISION :** 1-2 phrases maximum.
- **BOUNDING BOX :** Si le diagnostic s'applique à toute la parcelle: `{'x_min': 0.0, 'y_min': 0.0, 'x_max': 1.0, 'y_max': 1.0}`.
"""

DRONE_PROMPT = """
Tu es 'PestAI-DroneVision', un moteur d'analyse IA spécialisé dans les images de DRONE à très haute résolution (Orthophotos RVB et Multispectrales).

**TA MISSION :**
1.  **Identifier le Sujet Principal :** Toujours 'DRONE_PLOT'.
2.  **Mener une Analyse de Précision :**
    - Identifie les anomalies agronomiques (ex: 'Infestation de Mauvaises Herbes', 'Stress hydrique localisé', 'Faible densité de semis', 'Déficience en Azote').
    - Évalue la sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL') pour chaque anomalie.
    - Délimite la zone exacte de l'anomalie avec une `boundingBox`.
    - Génère des `knowledgeBaseTags`.
3.  **Fournir une Réponse JSON Strictement Structurée :** EXCLUSIVEMENT au format JSON.

{
  "subject": {
    "subjectType": "string (Toujours 'DRONE_PLOT')",
    "description": "string (ex: 'Analyse d'image drone.')",
    "confidence": 1.0
  },
  "detections": [
    {
      "className": "string (Nom de l'anomalie, ex: 'Infestation de Mauvaises Herbes')",
      "confidenceScore": "float",
      "severity": "string ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string (Description simple de l'anomalie)",
        "impact": "string (Impact simple sur la zone)",
        "recommendations": {
          "biological": [],
          "chemical": [ { "solution": "string", "details": "string", "source": null } ],
          "cultural": [ { "solution": "string", "details": "string", "source": null } ]
        },
        "knowledgeBaseTags": [ "string" ]
      }
    }
  ]
}

**RÈGLES D'OR v11 :**
- **LANGAGE (TRÈS IMPORTANT) :** Utilise des phrases simples pour le TTS Wolof.
- **CONCISION :** 1-2 phrases maximum.
- **BOUNDING BOX :** Si l'anomalie s'applique à toute l'image: `{'x_min': 0.0, 'y_min': 0.0, 'x_max': 1.0, 'y_max': 1.0}`.
"""

# --- Logique d'appel à l'IA avec rotation de clés ---
async def generate_gemini_analysis_with_key_rotation(
    prompt: str,
    image_part: dict,
    config: genai.types.GenerationConfig
):
    initial_key = key_manager.get_current_key()

    for _ in range(len(key_manager.keys)):
        try:
            current_key = key_manager.get_current_key()
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = await model.generate_content_async(
                [prompt, image_part],
                generation_config=config,
                request_options={'timeout': 120}
            )
            return response

        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.PermissionDenied) as e:
            print(f"Erreur de quota/permission pour la clé ...{current_key[-4:]}.")
            await key_manager.switch_to_next_key_async()
            if key_manager.get_current_key() == initial_key:
                print("Toutes les clés API ont été essayées et ont échoué.")
                raise HTTPException(status_code=429, detail="Toutes les clés API Gemini sont indisponibles ou ont dépassé leur quota.")

    raise HTTPException(status_code=503, detail="Échec de l'analyse IA après avoir essayé toutes les clés API disponibles.")


# --- Création de la clé de cache unifiée ---
def unified_key_builder(
    func,
    namespace: str = "",
    *,
    request: Request,
    response: Response,
    **kwargs
):
    analysis_type_str = str(kwargs.get("analysis_type", "unknown"))
    file: UploadFile = kwargs["file"]
    file_content = file.file.read()
    file.file.seek(0)
    file_hash = hashlib.sha256(file_content).hexdigest()
    return f"{namespace}:{analysis_type_str}:{file_hash}"


# --- Point d'Entrée Unifié v11 ---
@app.post(
    "/api/v11/analyze-unified",
    response_model=AIAnalysisResponse,
    summary="Analyse unifiée (Plante, Satellite, Drone) v11",
    tags=["IA Analysis Service v11"]
)
@limiter.limit("15/minute")
@cache(namespace="pestai-analysis", expire=86400, key_builder=unified_key_builder)
async def analyze_unified_endpoint(
    request: Request,
    response: Response,
    analysis_type: AnalysisType = Form(
        ...,
        description="Le type d'analyse: 'PLANT_PEST', 'SATELLITE_REMOTE_SENSING', ou 'DRONE_ANALYSIS'."
    ),
    file: UploadFile = File(
        ...,
        description="Fichier image (JPEG, JPG, PNG). Photo de proximité, image satellite ou orthophoto de drone."
    )
):
    """
    **Rôle de ce service v11 :**

    1.  Recevoir une image ET un type d'analyse (Plante, Satellite, Drone).
    2.  Sélectionner le prompt expert approprié (optimisé pour TTS/Wolof).
    3.  Appeler Gemini et gérer la rotation des clés.
    4.  Découper les 'détections' (lésions ou zones) et les uploader sur Cloudinary.
    5.  Retourner le JSON structuré `AIAnalysisResponse`.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/tiff"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG, JPG, PNG ou TIFF.")

    if analysis_type == AnalysisType.PLANT_PEST:
        selected_prompt = PLANT_PEST_PROMPT
    elif analysis_type == AnalysisType.SATELLITE_REMOTE_SENSING:
        selected_prompt = SATELLITE_PROMPT
    elif analysis_type == AnalysisType.DRONE_ANALYSIS:
        selected_prompt = DRONE_PROMPT
    else:
        raise HTTPException(status_code=400, detail="Type d'analyse inconnu.")

    image_bytes = await file.read()

    try:
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        gemini_response = await generate_gemini_analysis_with_key_rotation(
            prompt=selected_prompt,
            image_part=image_part,
            config=generation_config
        )

        analysis_data = json.loads(gemini_response.text)

        if analysis_data.get("detections") and file.content_type in ["image/jpeg", "image/png", "image/jpg"]:
            try:
                original_image = Image.open(io.BytesIO(image_bytes))
                width, height = original_image.size

                for detection in analysis_data["detections"]:
                    if "boundingBox" not in detection:
                        continue
                    bbox = detection["boundingBox"]
                    coords = (
                        int(bbox["x_min"] * width),
                        int(bbox["y_min"] * height),
                        int(bbox["x_max"] * width),
                        int(bbox["y_max"] * height)
                    )
                    cropped_image = original_image.crop(coords)
                    buffer = io.BytesIO()
                    cropped_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    upload_result = cloudinary.uploader.upload(buffer, folder="pestai_detections_v11")
                    detection["croppedImageUrl"] = upload_result.get("secure_url")
            except Exception as e:
                print(f"Avertissement: Échec du découpage de l'image. Erreur: {str(e)}")

        return analysis_data

    except json.JSONDecodeError:
        error_text = gemini_response.text if 'gemini_response' in locals() else "Pas de réponse de l'IA"
        raise HTTPException(status_code=502, detail=f"Réponse invalide du service d'IA (non-JSON): {error_text}")

    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=503, detail=f"Erreur non gérée de l'API Google : {e.message}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne inattendue du serveur : {str(e)}")


# --- Point de Santé ---
@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "PestAI - Unified Analysis Microservice v11.0.0 est opérationnel."}
