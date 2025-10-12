import os
import json
import hashlib
import asyncio
import io # Ajouté pour la manipulation d'images en mémoire
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions

# --- Dépendances pour le traitement d'image et Cloudinary ---
import cloudinary
import cloudinary.uploader
from PIL import Image # Pillow

# --- Configuration Initiale ---
load_dotenv()

# Configuration Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Clé API Gemini non trouvée.")
genai.configure(api_key=GEMINI_API_KEY)

# Configuration Cloudinary
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
     raise ValueError("Les configurations Cloudinary sont manquantes.")
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# --- Configuration du Rate Limiter ---
limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])

# --- Initialisation de l'Application FastAPI ---
app = FastAPI(
    title="PestAI - IA Analysis Microservice",
    description="API v8.4. Service d'analyse d'images optimisé avec cropping des détections et upload sur Cloudinary.",
    version="8.4.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Structures de Données Pydantic (Mise à jour) ---
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
    # CHAMP AJOUTÉ : L'URL de l'image découpée
    croppedImageUrl: Optional[str] = Field(None, description="URL de l'image découpée montrant la détection.")

class AnalysisSubject(BaseModel):
    subjectType: str
    description: str
    confidence: float

class AIAnalysisResponse(BaseModel):
    subject: AnalysisSubject
    detections: List[Detection]

# --- Le Prompt (Inchangé) ---
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

# --- Logique d'appel à l'IA avec gestion des erreurs et des réessais ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        asyncio.TimeoutError
    ))
)
async def generate_gemini_analysis(image_part: dict, config: genai.types.GenerationConfig):
    """Appelle l'API Gemini avec une politique de réessai intelligente."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = await model.generate_content_async(
        [UNIVERSAL_PROMPT, image_part],
        generation_config=config,
        request_options={'timeout': 120}
    )
    return response

# --- Création de la clé de cache ---
def image_key_builder(func, namespace: str = "", *, request: Request, response: Response, **kwargs):
    file_content = kwargs["file"].file.read()
    kwargs["file"].file.seek(0)
    file_hash = hashlib.sha256(file_content).hexdigest()
    return f"{namespace}:{file_hash}"


# --- Le Point d'Entrée (Endpoint) du Microservice Mis à Jour ---
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
    - Retourne un JSON pur et validé.
    - **Nouveau :** Pour chaque détection, découpe l'image originale, l'envoie sur Cloudinary
      et ajoute l'URL de l'image découpée dans la réponse.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG ou PNG.")

    # Lire l'image en mémoire une seule fois pour la réutiliser
    image_bytes = await file.read()

    try:
        # --- Étape 1: Analyse par l'IA (inchangée) ---
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        gemini_response = await generate_gemini_analysis(image_part, generation_config)
        analysis_data = json.loads(gemini_response.text)

        # --- Étape 2: Découpage et Upload des détections ---
        if analysis_data.get("detections"):
            original_image = Image.open(io.BytesIO(image_bytes))
            width, height = original_image.size

            for detection in analysis_data["detections"]:
                bbox = detection["boundingBox"]
                
                # Convertir les coordonnées normalisées en pixels
                left = int(bbox["x_min"] * width)
                top = int(bbox["y_min"] * height)
                right = int(bbox["x_max"] * width)
                bottom = int(bbox["y_max"] * height)

                # Découper l'image
                cropped_image = original_image.crop((left, top, right, bottom))
                
                # Sauvegarder l'image découpée dans un buffer en mémoire
                buffer = io.BytesIO()
                cropped_image.save(buffer, format="PNG")
                buffer.seek(0)
                
                # Uploader sur Cloudinary
                upload_result = cloudinary.uploader.upload(
                    buffer,
                    folder="pestai_detections", # Optionnel: pour organiser dans Cloudinary
                    public_id=f"detection_{hashlib.sha1(image_bytes).hexdigest()}_{detection['className']}"
                )
                
                # Ajouter l'URL à la détection
                detection["croppedImageUrl"] = upload_result.get("secure_url")

        return analysis_data

    except json.JSONDecodeError:
        error_text = gemini_response.text if 'gemini_response' in locals() else "No response"
        raise HTTPException(status_code=502, detail=f"Réponse invalide du service d'IA (non-JSON): {error_text}")
    
    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=503, detail=f"Erreur du service IA (API Google): {e.message}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors du traitement: {str(e)}")


# --- Événements de Démarrage et Point de Santé ---
@app.on_event("startup")
async def startup():
    """Initialise le cache en mémoire au démarrage de l'application."""
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache en mémoire initialisé. Le service est prêt à analyser.")

@app.get("/", include_in_schema=False)
def read_root():
    """Endpoint de santé pour vérifier que le service est en ligne."""
    return {"message": "PestAI - IA Analysis Microservice v8.4 est opérationnel."}
