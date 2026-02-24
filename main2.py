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
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
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
    Une classe pour gérer une liste de clés API et permettre de passer
    à la suivante en cas d'erreur de quota.
    """
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

# --- Chargement et configuration des clés Gemini ---
gemini_api_keys_str = os.getenv("GEMINI_API_KEYS")
if not gemini_api_keys_str:
    raise ValueError("Variable d'environnement GEMINI_API_KEYS non trouvée. Assure-toi qu'elle est dans le fichier .env.")
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
    print("Cache en mémoire initialisé. Le service est prêt à analyser.")
    yield


# --- Initialisation de l'Application FastAPI ---
app = FastAPI(
    title="PestAI - IA Analysis Microservice",
    description="API v8.5. Service d'analyse robuste avec rotation automatique des clés API Gemini.",
    version="8.5.0",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
    croppedImageUrl: Optional[str] = Field(None, description="URL de l'image découpée montrant la détection.")

class AnalysisSubject(BaseModel):
    subjectType: str
    description: str
    confidence: float

class AIAnalysisResponse(BaseModel):
    subject: AnalysisSubject
    detections: List[Detection]

# --- Prompt Universel ---
UNIVERSAL_PROMPT = """
Tu es 'PestAI-Core', un moteur d'analyse d'images agronomiques de classe mondiale. Ta seule fonction est de recevoir une image et de retourner une analyse experte complète, structurée et riche en données.

**TA MISSION :**
1.  **Identifier le Sujet Principal :** Détermine si le sujet est une 'PLANT', un 'PEST', ou 'UNKNOWN'.
2.  **Mener une Analyse Complète :**
    - Identifie l'espèce du sujet et chaque problème (maladie/ravageur).
    - Pour chaque détection, tu DOIS évaluer sa sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    - Pour chaque détection, tu DOIS générer des mots-clés pertinents (`knowledgeBaseTags`).
3.  **Fournir une Réponse JSON Strictement Structurée :** Ta réponse doit être EXCLUSIVEMENT au format JSON. Ne renvoie AUCUN texte avant ou après. Le schéma est le suivant :

{
  "subject": {
    "subjectType": "string ('PLANT', 'PEST', or 'UNKNOWN')",
    "description": "string (ex: 'Plant de Maïs (Zea mays)')",
    "confidence": "float (0.0-1.0)"
  },
  "detections": [
    {
      "className": "string (Nom du problème)",
      "confidenceScore": "float",
      "severity": "string (Choisis parmi 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string (Description détaillée)",
        "impact": "string (Impact sur les cultures)",
        "recommendations": {
          "biological": [ { "solution": "string", "details": "string", "source": "string (URL)" } ],
          "chemical": [ { "solution": "string", "details": "string", "source": "string (URL)" } ],
          "cultural": [ { "solution": "string", "details": "string", "source": "string (URL)" } ]
        },
        "knowledgeBaseTags": [ "string (Liste de mots-clés pertinents pour la recherche)" ]
      }
    }
  ]
}

**RÈGLES D'OR :**
- **SÉVÉRITÉ OBLIGATOIRE :** Le champ `severity` est crucial.
- **TAGS OBLIGATOIRES :** Le champ `knowledgeBaseTags` doit être fourni.
- **GROUPEMENT & SOURÇAGE :** Les recommandations DOIVENT être groupées et sourcées. Si une catégorie est vide, renvoie un tableau vide.
- **NORMALISATION :** Les coordonnées de `boundingBox` DOIVENT être normalisées (0.0 à 1.0).
"""

# --- Logique d'appel à l'IA avec rotation de clés ---
async def generate_gemini_analysis_with_key_rotation(image_part: dict, config: genai.types.GenerationConfig):
    """
    Tente d'appeler l'API Gemini. Si une erreur de quota survient,
    bascule vers la clé suivante jusqu'à avoir essayé toutes les clés.
    """
    initial_key = key_manager.get_current_key()

    for _ in range(len(key_manager.keys)):
        try:
            current_key = key_manager.get_current_key()
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = await model.generate_content_async(
                [UNIVERSAL_PROMPT, image_part],
                generation_config=config,
                request_options={'timeout': 120}
            )
            return response

        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.PermissionDenied) as e:
            print(f"Erreur de quota ou de permission pour la clé ...{current_key[-4:]}.")
            key_manager.switch_to_next_key()
            if key_manager.get_current_key() == initial_key:
                print("Toutes les clés API ont été essayées et ont échoué.")
                raise HTTPException(status_code=429, detail="Toutes les clés API Gemini sont indisponibles ou ont dépassé leur quota.")

    raise HTTPException(status_code=503, detail="Échec de l'analyse IA après avoir essayé toutes les clés API disponibles.")


# --- Création de la clé de cache ---
def image_key_builder(func, namespace: str = "", *, request: Request, response: Response, **kwargs):
    file_content = kwargs["file"].file.read()
    kwargs["file"].file.seek(0)
    file_hash = hashlib.sha256(file_content).hexdigest()
    return f"{namespace}:{file_hash}"


# --- Point d'Entrée du Microservice ---
@app.post(
    "/api/v8/analyze-image",
    response_model=AIAnalysisResponse,
    summary="Prend une image, retourne une analyse IA et upload les détections.",
    tags=["IA Analysis Service"]
)
@limiter.limit("15/minute")
@cache(namespace="pestai-analysis", expire=86400, key_builder=image_key_builder)
async def analyze_image_endpoint(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="Fichier image (JPEG, PNG) de la plante ou du ravageur."),
):
    """
    **Rôle de ce service :**
    - Retourne un JSON pur et validé, prêt à être consommé.
    - Pour chaque détection, découpe l'image originale, l'envoie sur Cloudinary et
      ajoute l'URL de l'image découpée dans la réponse.
    - Gère les pannes de clés API en basculant automatiquement vers la suivante.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG ou PNG.")

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
                upload_result = cloudinary.uploader.upload(buffer, folder="pestai_detections")
                detection["croppedImageUrl"] = upload_result.get("secure_url")

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
    return {"message": "PestAI - IA Analysis Microservice v8.5 est opérationnel."}
