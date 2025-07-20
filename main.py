
import os
import json
import hashlib
import asyncio
from enum import Enum
from typing import List, Optional

# --- Dépendances ---
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Dépendances pour l'optimisation des quotas (sans Redis) ---
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import google.api_core.exceptions

# --- Configuration Initiale ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Clé API Gemini non trouvée. Veuillez la définir dans la variable d'environnement GEMINI_API_KEY.")

genai.configure(api_key=GEMINI_API_KEY)

# --- Configuration du Rate Limiter (Protection Anti-Burst) ---
# Limite les requêtes par adresse IP à 15 par minute. Fonctionne en mémoire.
limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])


# --- Initialisation de l'Application FastAPI ---
app = FastAPI(
    title="PestAI - IA Analysis Microservice",
    description="API v8.3. Service d'analyse d'images optimisé avec un cache en mémoire, du rate limiting et des réessais pour préserver les quotas d'API.",
    version="8.3.0"
)
# Intégration du Rate Limiter dans l'application
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Structures de Données Pydantic (Le "Contrat" de données) ---
# Ce bloc est inchangé, il définit la structure de la réponse.

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

class AnalysisSubject(BaseModel):
    subjectType: str
    description: str
    confidence: float

class AIAnalysisResponse(BaseModel):
    subject: AnalysisSubject
    detections: List[Detection]


# --- Le Prompt ---
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
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = await model.generate_content_async(
        [UNIVERSAL_PROMPT, image_part],
        generation_config=config,
        request_options={'timeout': 120}
    )
    return response

# --- Création de la clé de cache à partir du contenu de l'image (STRATÉGIE #1) ---
def image_key_builder(func, namespace: str = "", *, request: Request, response: Response, **kwargs):
    """Crée une clé de cache unique en hashant le contenu binaire de l'image."""
    file_content = kwargs["file"].file.read()
    kwargs["file"].file.seek(0)
    file_hash = hashlib.sha256(file_content).hexdigest()
    return f"{namespace}:{file_hash}"


# --- Le Point d'Entrée (Endpoint) du Microservice Optimisé ---
@app.post(
    "/api/v8/analyze-image",
    response_model=AIAnalysisResponse,
    summary="Prend une image et retourne une analyse IA complète.",
    tags=["IA Analysis Service"]
)
@limiter.limit("15/minute")  # Protection anti-burst (STRATÉGIE #2)
@cache(namespace="pestai-analysis", expire=86400, key_builder=image_key_builder)  # Cache le résultat 24h (STRATÉGIE #1)
async def analyze_image_endpoint(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="Fichier image (JPEG, PNG) de la plante ou du ravageur."),
):
    """
    **Rôle de ce service :**
    - Retourne un JSON pur et validé, prêt à être consommé par le backend principal.
    - **Optimisations :** Le résultat pour chaque image unique est mis en cache en mémoire pendant 24h
      pour économiser le quota d'API. Le rate limiting prévient les abus.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG ou PNG.")

    gemini_response = None
    try:
        image_bytes = await file.read()
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        
        # Appel à la fonction encapsulée avec gestion des erreurs et réessais (STRATÉGIE #3)
        gemini_response = await generate_gemini_analysis(image_part, generation_config)

        analysis_json = json.loads(gemini_response.text)
        return analysis_json

    except json.JSONDecodeError:
        error_text = gemini_response.text if gemini_response else "No response"
        print(f"Erreur de parsing JSON critique. Texte reçu: {error_text}")
        raise HTTPException(status_code=502, detail="Réponse invalide du service d'IA (non-JSON).")
    
    except google.api_core.exceptions.GoogleAPICallError as e:
        print(f"Erreur API Google non récupérable: {e}")
        if isinstance(e, google.api_core.exceptions.PermissionDenied):
             raise HTTPException(status_code=403, detail="Erreur d'authentification avec l'API Gemini. Vérifiez la clé API.")
        if isinstance(e, google.api_core.exceptions.InvalidArgument):
             raise HTTPException(status_code=400, detail=f"Argument invalide envoyé à l'API Gemini. Détail: {e.message}")
        raise HTTPException(status_code=503, detail=f"Erreur du service IA (API Google): {e.message}")

    except Exception as e:
        print(f"Erreur inattendue durant l'analyse: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {str(e)}")


# --- Événements de Démarrage et Point de Santé ---
@app.on_event("startup")
async def startup():
    """Initialise le cache en mémoire au démarrage de l'application."""
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache en mémoire initialisé. Le service est prêt à analyser.")

@app.get("/", include_in_schema=False)
def read_root():
    """Endpoint de santé pour vérifier que le service est en ligne."""
    return {"message": "PestAI - IA Analysis Microservice v8.3 est opérationnel."}
