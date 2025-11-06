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
# MODIFIÉ: 'Form' n'est plus nécessaire
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

# --- GESTIONNAIRE DE CLÉS API GEMINI (Inchangé - v2 Optimisée) ---
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

# --- Configuration des clés Gemini (inchangée) ---
gemini_api_keys_str = os.getenv("GEMINI_API_KEYS")
if not gemini_api_keys_str:
    raise ValueError("Variable d'environnement GEMINI_API_KEYS non trouvée.")
gemini_api_keys = [key.strip() for key in gemini_api_keys_str.split(',') if key.strip()]
key_manager = KeyManager(gemini_api_keys)

# --- Configuration Cloudinary (inchangée) ---
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# --- Configuration du Rate Limiter (inchangée) ---
limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])

# --- Initialisation de l'Application FastAPI (MODIFIÉE) ---
app = FastAPI(
    title="PestAI - Remote Sensing Service",
    description="API v1.0. Service dédié à l'analyse par télédétection d'images satellites.",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- Structures de Données Pydantic (Inchangées) ---
# La structure AIAnalysisResponse est parfaitement adaptée.
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

# --- MODIFIÉ: Prompt Unique pour la Télédétection ---
REMOTE_SENSING_PROMPT = """
Tu es 'PestAI-RemoteSensing', un moteur d'analyse expert en agronomie et en télédétection. Ta fonction est l'analyse d'images SATELLITES (ex: Sentinel-2, Landsat) de parcelles agricoles.
Tu es spécifiquement calibré pour l'agriculture en Afrique de l'Ouest (Sénégal, Zone des Niayes, Vallée du Fleuve Sénégal).

**TA MISSION :**
1.  **Identifier le Sujet Principal :** Toujours 'SATELLITE_PLOT'. La description doit être 'Analyse de parcelle agricole'.
2.  **Mener une Analyse Diagnostique Zonale :**
    - Identifie les problèmes agronomiques majeurs visibles depuis l'espace (Stress hydrique, Vigueur de la végétation (basée sur NDVI implicite), Déficience en nutriments (ex: Azote), possible Salinisation du sol).
    - Pour chaque diagnostic zonal, tu DOIS évaluer sa sévérité ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
    - Pour chaque diagnostic, tu DOIS délimiter la zone affectée avec une `boundingBox`.
    - Pour chaque diagnostic, tu DOIS générer des mots-clés (`knowledgeBaseTags`).
3.  **Fournir une Réponse JSON Strictement Structurée :** Ta réponse doit être EXCLUSIVEMENT au format JSON. Ne renvoie AUCUN texte avant ou après. Le schéma est le suivant :

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

# --- MODIFIÉ : Logique d'appel à l'IA (simplifiée) ---
async def generate_gemini_analysis_with_key_rotation(
    image_part: dict,
    config: genai.types.GenerationConfig
):
    """
    Tente d'appeler l'API Gemini avec le prompt de TÉLÉDÉTECTION.
    Gère la rotation des clés en cas d'erreur de quota.
    """
    initial_key = key_manager.get_current_key()
    
    for _ in range(len(key_manager.keys)):
        try:
            current_key = key_manager.get_current_key()
            genai.configure(api_key=current_key)
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = await model.generate_content_async(
                # MODIFIÉ: Le prompt est maintenant "en dur"
                [REMOTE_SENSING_PROMPT, image_part],
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


# --- MODIFIÉ : Création de la clé de cache (simplifiée) ---
def image_hash_key_builder(
    func,
    namespace: str = "",
    *,
    request: Request,
    response: Response,
    **kwargs
):
    """ Crée une clé de cache unique basée sur le hash de l'image. """
    file: UploadFile = kwargs["file"]
    file_content = file.file.read()
    file.file.seek(0)  # Rembobine le fichier pour la suite
    file_hash = hashlib.sha256(file_content).hexdigest()
    
    return f"{namespace}:{file_hash}"


# --- MODIFIÉ : Le Point d'Entrée (Endpoint) Dédié ---
@app.post(
    "/api/v1/analyze-satellite",
    response_model=AIAnalysisResponse,
    summary="Analyse une image satellite pour un diagnostic agronomique.",
    tags=["Remote Sensing Service"]
)
@limiter.limit("15/minute")
@cache(namespace="remotesensing-analysis", expire=86400, key_builder=image_hash_key_builder)
async def analyze_satellite_image(
    request: Request,
    response: Response,
    file: UploadFile = File(
        ...,
        description="Image satellite (JPEG, JPG, PNG) d'une parcelle agricole."
    )
):
    """
    **Rôle de ce service dédié :**
    
    1.  **Recevoir une image satellite.**
    2.  **Appeler Gemini** avec le prompt de télédétection.
    3.  **Gérer la rotation des clés API** en cas de panne.
    4.  **Découper les 'détections'** (zones de stress) et les uploader.
    5.  **Retourner le JSON** structuré `AIAnalysisResponse`.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG, JPG ou PNG.")

    image_bytes = await file.read()

    try:
        # --- Appel de l'IA (simplifié) ---
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        
        gemini_response = await generate_gemini_analysis_with_key_rotation(
            image_part=image_part,
            config=generation_config
        )

        analysis_data = json.loads(gemini_response.text)

        # --- Logique de découpage (inchangée, elle est universelle) ---
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
                    # MODIFIÉ: Dossier Cloudinary dédié
                    folder="pestai_satellite_detections"
                )
                
                detection["croppedImageUrl"] = upload_result.get("secure_url")

        # --- Retour de la réponse validée ---
        return analysis_data

    except json.JSONDecodeError:
        error_text = gemini_response.text if 'gemini_response' in locals() else "Pas de réponse de l'IA"
        raise HTTPException(status_code=502, detail=f"Réponse invalide du service d'IA (non-JSON): {error_text}")
    
    except google.api_core.exceptions.GoogleAPICallError as e:
        raise HTTPException(status_code=503, detail=f"Erreur non gérée de l'API Google : {e.message}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne inattendue du serveur : {str(e)}")


# --- Événements de Démarrage et Point de Santé (MODIFIÉS) ---
@app.on_event("startup")
async def startup():
    """Initialise le cache en mémoire au démarrage de l'application."""
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache en mémoire initialisé. Service de Télédétection v1.0.0 prêt.")

@app.get("/", include_in_schema=False)
def read_root():
    """Endpoint de santé pour vérifier que le service est en ligne."""
    return {"message": "PestAI - Remote Sensing Service v1.0.0 est opérationnel."}
