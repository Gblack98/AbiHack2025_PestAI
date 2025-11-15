import os
import json
import hashlib
import asyncio
import io
import itertools
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

# --- Initialisation de l'Application FastAPI  ---
app = FastAPI(
    title="PestAI - Unified Analysis Microservice",
    description="API v10.0. Service unifié pour l'analyse multi-échelle : Plante, Satellite ET Drone.",
    version="10.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Types d'analyse possibles ---
class AnalysisType(str, Enum):
    PLANT_PEST = "PLANT_PEST"
    SATELLITE_REMOTE_SENSING = "SATELLITE_REMOTE_SENSING"
    DRONE_ANALYSIS = "DRONE_ANALYSIS" 

# --- Structures de Données Pydantic (Inchangées) ---
# Notre structure AIAnalysisResponse est universelle.
# Pour le Drone :
# - Detection.className = "Infestation de Mauvaises Herbes"
# - Detection.severity = "MEDIUM"
# - Detection.boundingBox = Coordonnées de la zone infestée
# - Detection.details.recommendations = { "chemical": [{ "solution": "Herbispray ciblé" ... }] }

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

# --- NOUVEAU: Banque de Prompts ---

# PROMPT 1: Analyse de Proximité (Plantes & Ravageurs) - 
PLANT_PEST_PROMPT = """
Tu es 'PestAI-Core', un moteur d'analyse d'images agronomiques de classe mondiale. Ta fonction est l'analyse d'images de PROXIMITÉ (feuilles, tiges, insectes).

**TA MISSION :**
1.  **Identifier le Sujet Principal :** 'PLANT', 'PEST', ou 'UNKNOWN'.
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
      "className": "string (Nom du problème, ex: 'Rouille commune du maïs')",
      "confidenceScore": "float",
      "severity": "string ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string (Description détaillée du problème)",
        "impact": "string (Impact sur la plante)",
        "recommendations": {
          "biological": [ { "solution": "string", "details": "string", "source": "string (URL)" } ],
          "chemical": [ { "solution": "string", "details": "string", "source": "string (URL)" } ],
          "cultural": [ { "solution": "string", "details": "string", "source": "string (URL)" } ]
        },
        "knowledgeBaseTags": [ "string (Liste de mots-clés pertinents)" ]
      }
    }
  ]
}

**RÈGLES D'OR (PROXIMITÉ) :**
- **SÉVÉRITÉ OBLIGATOIRE :** Le champ `severity` est crucial.
- **BOUNDING BOX PRÉCISE :** Les boîtes doivent cibler la lésion ou le ravageur.
- **NORMALISATION :** Les coordonnées de `boundingBox` DOIVENT être normalisées (0.0 à 1.0).
"""

# PROMPT 2: Analyse Satellite (Télédétection) 
SATELLITE_PROMPT = """
Tu es 'PestAI-RemoteSensing', un moteur d'analyse expert en agronomie et en télédétection. Ta fonction est l'analyse d'images SATELLITES (ex: Sentinel-2, Landsat) de parcelles agricoles.
Tu es spécifiquement calibré pour l'agriculture en Afrique de l'Ouest (Sénégal, Zone des Niayes, Vallée du Fleuve Sénégal).

**TA MISSION :**
1.  **Identifier le Sujet Principal :** Toujours 'SATELLITE_PLOT'. La description doit être 'Analyse de parcelle agricole'.
2.  **Mener une Analyse Diagnostique Zonale :**
    - Identifie les problèmes agronomiques majeurs visibles depuis l'espace (Stress hydrique, Vigueur de la végétation (basée sur NDVI implicite), Déficience en nutriments (ex: Azote), possible Salinisation du sol).
    - Pour chaque diagnostic zonal, tu DOIS évaluer sa sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    - Pour chaque diagnostic, tu DOIS délimiter la zone affectée avec une `boundingBox`.
    - Pour chaque diagnostic, tu DOIS générer des mots-clés (`knowledgeBaseTags`).
3.  **Fournir une Réponse JSON Strictement Structurée :** Ta réponse doit être EXCLUSIVEMENT au format JSON. Ne renvoie AUCUN texte avant ou après. Le schéma est IDENTIQUE à l'analyse de proximité :

{
  "subject": {
    "subjectType": "string (Toujours 'SATELLITE_PLOT')",
    "description": "string (ex: 'Analyse de parcelle agricole, région de Saint-Louis')",
    "confidence": "float (Toujours 1.0)"
  },
  "detections": [
    {
      "className": "string (Nom du diagnostic, ex: 'Stress Hydrique', 'Faible vigueur de végétation', 'Déficience en Azote suspectée')",
      "confidenceScore": "float",
      "severity": "string ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string (Description détaillée du diagnostic zonal)",
        "impact": "string (Impact sur le rendement potentiel de la parcelle)",
        "recommendations": {
          "biological": [],
          "chemical": [ { "solution": "Application d'engrais azoté", "details": "Appliquer X unités d'azote...", "source": null } ],
          "cultural": [ { "solution": "Optimisation de l'irrigation", "details": "Augmenter l'apport en eau de 20% sur la zone...", "source": null } ]
        },
        "knowledgeBaseTags": [ "string (ex: 'Stress Hydrique', 'NDVI', 'Sentinel-2', 'Gestion Eau', 'Fertilisation')" ]
      }
    }
  ]
}

**RÈGLES D'OR (SATELLITE) :**
- **ANALYSE ZONALE :** Les 'détections' sont des 'diagnostics zonaux'.
- **RECOMMANDATIONS PRAGMATIQUES :** Les recommandations doivent être actionnables (irrigation, fertilisation, etc.).
- **NORMALISATION :** Les coordonnées de `boundingBox` DOIVENT être normalisées (0.0 à 1.0).
"""

# NOUVEAU PROMPT 3: Analyse par Drone
DRONE_PROMPT = """
Tu es 'PestAI-DroneVision', un moteur d'analyse IA spécialisé dans les images de DRONE à très haute résolution (Orthophotos RVB et Multispectrales).
Tu es calibré pour l'agriculture de précision au Sénégal.

**TA MISSION :**
1.  **Identifier le Sujet Principal :** Toujours 'DRONE_PLOT'. La description doit être 'Analyse orthophoto de parcelle'.
2.  **Mener une Analyse de Précision :**
    - Identifie les anomalies agronomiques visibles (ex: 'Infestation de Mauvaises Herbes', 'Stress hydrique localisé', 'Faible densité de semis', 'Déficience en Azote').
    - Pour chaque anomalie, tu DOIS évaluer sa sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    - Pour chaque anomalie, tu DOIS délimiter la zone exacte avec une `boundingBox`.
    - Pour chaque anomalie, tu DOIS générer des mots-clés (`knowledgeBaseTags`).
3.  **Fournir une Réponse JSON Strictement Structurée :** Ta réponse doit être EXCLUSIVEMENT au format JSON. Le schéma est IDENTIQUE aux autres analyses :

{
  "subject": {
    "subjectType": "string (Toujours 'DRONE_PLOT')",
    "description": "string (ex: 'Analyse orthophoto de parcelle, 25 Hectares')",
    "confidence": "float (Toujours 1.0)"
  },
  "detections": [
    {
      "className": "string (Nom de l'anomalie, ex: 'Infestation de Mauvaises Herbes', 'Stress hydrique localisé')",
      "confidenceScore": "float",
      "severity": "string ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
      "boundingBox": { "x_min": "float", "y_min": "float", "x_max": "float", "y_max": "float" },
      "details": {
        "description": "string (Description détaillée de l'anomalie détectée)",
        "impact": "string (Impact sur la zone : concurrence pour les ressources, perte de rendement localisée...)",
        "recommendations": {
          "biological": [],
          "chemical": [ { "solution": "Application ciblée d'herbicide", "details": "Appliquer [Produit X] uniquement sur les zones détectées...", "source": null } ],
          "cultural": [ { "solution": "Modulation de l'irrigation", "details": "Augmenter l'apport en eau de 15% sur cette zone...", "source": null } ]
        },
        "knowledgeBaseTags": [ "string (ex: 'Drone', 'Orthophoto', 'Mauvaises Herbes', 'Agriculture de Précision')" ]
      }
    }
  ]
}

**RÈGLES D'OR (DRONE) :**
- **ANALYSE MICRO-ZONALE :** Les 'détections' sont des anomalies très localisées.
- **RECOMMANDATIONS CIBLÉES :** Les recommandations doivent être adaptées à l'agriculture de précision (ex: modulation, application ciblée).
- **NORMALISATION :** Les coordonnées de `boundingBox` DOIVENT être normalisées (0.0 à 1.0).
"""

# --- Logique d'appel à l'IA  ---
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


# --- Création de la clé de cache unifiée  ---
# Elle prend déjà en compte "analysis_type", donc elle fonctionnera
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


@app.post(
    "/api/v10/analyze-unified", 
    response_model=AIAnalysisResponse,
    summary="Analyse unifiée (Plante/Ravageur, Satellite OU Drone).",
    tags=["IA Analysis Service v10"]
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
        description="Fichier image (JPEG,JPG, PNG). Photo de proximité, image satellite ou orthophoto de drone."
    )
):
    """
    **Rôle de ce service unifié v10 :**
    
    1.  Recevoir une image ET un type d'analyse (Plante, Satellite, Drone).
    2.  Sélectionner le prompt expert approprié.
    3.  Appeler Gemini et gérer la rotation des clés.
    4.  Découper les 'détections' (lésions ou zones) et les uploader sur Cloudinary.
    5.  Retourner le JSON structuré `AIAnalysisResponse`.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG, JPG ou PNG.")

    # --- 1. Sélection du Prompt  ---
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
        # --- 2. Appel de l'IA ---
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        
        gemini_response = await generate_gemini_analysis_with_key_rotation(
            prompt=selected_prompt,
            image_part=image_part,
            config=generation_config
        )

        analysis_data = json.loads(gemini_response.text)

        # --- 3. Logique de découpage (Inchangée) ---
        if analysis_data.get("detections"):
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
                
                upload_result = cloudinary.uploader.upload(
                    buffer,
                    folder="pestai_detections_v10" 
                )
                
                detection["croppedImageUrl"] = upload_result.get("secure_url")

        # --- 4. Retour de la réponse validée ---
        return analysis_data

    except json.JSONDecodeError:
        error_text = gemini_response.text if 'gemini_response' in locals() else "Pas de réponse de l'IA"
        raise HTTPException(status_code=502, detail=f"Réponse invalide du service d'IA (non-JSON): {error_text}")
    
    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=503, detail=f"Erreur non gérée de l'API Google : {e.message}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne inattendue du serveur : {str(e)}")


# --- Événements de Démarrage et Point de Santé  ---
@app.on_event("startup")
async def startup():
    """Initialise le cache en mémoire au démarrage de l'application."""
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache en mémoire initialisé. Service unifié v10.0.0 prêt.")

@app.get("/", include_in_schema=False)
def read_root():
    """Endpoint de santé pour vérifier que le service est en ligne."""
    return {"message": "PestAI - Unified Analysis Microservice v10.0.0 est opérationnel."}
