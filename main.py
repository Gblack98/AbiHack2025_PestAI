import os
import json
import hashlib
import asyncio  # Ajouté pour le Lock
import io
import itertools
from enum import Enum
from typing import List, Optional

# --- Dépendances ---
import google.generativeai as genai
# MODIFIÉ: Ajout de Form pour le nouveau paramètre d'endpoint
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, Form
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Dépendances pour l'optimisation des quotas ---
from fastapi_cache import FastAPCache
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

# --- GESTIONNAIRE DE CLÉS API GEMINI (v2 - Optimisé avec asyncio.Lock) ---
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
        # NOUVEAU: Verrou pour gérer l'accès concurrentiel à la clé
        self._lock = asyncio.Lock()
        print(f"Gestionnaire de clés initialisé avec {len(self.keys)} clé(s).")

    def get_current_key(self) -> str:
        """Retourne la clé actuellement active."""
        return self._current_key

    async def switch_to_next_key_async(self) -> str:
        """
        Passe à la clé API suivante de manière asynchrone et sécurisée.
        Le verrou empêche 5 requêtes concurrentes d'utiliser 5 clés.
        """
        async with self._lock:
            # On vérifie la clé actuelle *avant* de la changer
            key_before_switch = self._current_key
            self._current_key = next(self._key_iterator)
            
            # On logue seulement si *cette* coroutine a effectivement changé la clé
            # (pour éviter les logs multiples si plusieurs attendent le verrou)
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
    title="PestAI - Unified Analysis Microservice",
    description="API v9.0.0. Service unifié pour l'analyse de plantes/ravageurs ET la télédétection satellite.",
    version="9.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- NOUVEAU: Types d'analyse possibles ---
class AnalysisType(str, Enum):
    PLANT_PEST = "PLANT_PEST"
    SATELLITE_REMOTE_SENSING = "SATELLITE_REMOTE_SENSING"

# --- Structures de Données Pydantic (Inchangées) ---
# La structure est assez générique pour s'adapter aux deux cas d'usage.
# Pour la télédétection:
# - Detection.className = "Stress Hydrique"
# - Detection.severity = "HIGH"
# - Detection.boundingBox = Coordonnées de la zone de stress
# - Detection.details.recommendations = { "cultural": [{ "solution": "Irrigation ciblée" ... }] }

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

# PROMPT 1: Analyse de Proximité (Plantes & Ravageurs)
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

# --- MODIFIÉ : Logique d'appel à l'IA avec rotation de clés ---
async def generate_gemini_analysis_with_key_rotation(
    prompt: str,  # NOUVEAU: Le prompt est maintenant un paramètre
    image_part: dict,
    config: genai.types.GenerationConfig
):
    """
    Tente d'appeler l'API Gemini avec un prompt spécifique.
    Gère la rotation des clés en cas d'erreur de quota de manière asynchrone.
    """
    initial_key = key_manager.get_current_key()
    
    for _ in range(len(key_manager.keys)):
        try:
            current_key = key_manager.get_current_key()
            genai.configure(api_key=current_key)
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = await model.generate_content_async(
                [prompt, image_part],  # Utilise le prompt fourni
                generation_config=config,
                request_options={'timeout': 120}
            )
            return response

        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.PermissionDenied) as e:
            print(f"Erreur de quota/permission pour la clé ...{current_key[-4:]}.")
            # MODIFIÉ: Appel de la version asynchrone sécurisée
            await key_manager.switch_to_next_key_async()
            
            if key_manager.get_current_key() == initial_key:
                print("Toutes les clés API ont été essayées et ont échoué.")
                raise HTTPException(status_code=429, detail="Toutes les clés API Gemini sont indisponibles ou ont dépassé leur quota.")
    
    raise HTTPException(status_code=503, detail="Échec de l'analyse IA après avoir essayé toutes les clés API disponibles.")


# --- NOUVEAU : Création de la clé de cache unifiée ---
def unified_key_builder(
    func,
    namespace: str = "",
    *,
    request: Request,
    response: Response,
    **kwargs
):
    """
    Crée une clé de cache unique basée sur le hash de l'image
    ET le type d'analyse demandé.
    """
    # Récupère le type d'analyse depuis les arguments de l'endpoint
    analysis_type_str = str(kwargs.get("analysis_type", "unknown"))
    
    # Récupère le hash du fichier (identique à avant)
    file: UploadFile = kwargs["file"]
    file_content = file.file.read()
    file.file.seek(0)  # Rembobine le fichier pour qu'il puisse être lu par la logique principale
    file_hash = hashlib.sha256(file_content).hexdigest()
    
    # Clé composite
    return f"{namespace}:{analysis_type_str}:{file_hash}"


# --- MODIFIÉ : Le Point d'Entrée (Endpoint) Unifié ---
@app.post(
    "/api/v9/analyze-unified",
    response_model=AIAnalysisResponse,
    summary="Analyse unifiée (Plante/Ravageur OU Satellite).",
    tags=["IA Analysis Service v9"]
)
@limiter.limit("15/minute")
@cache(namespace="pestai-analysis", expire=86400, key_builder=unified_key_builder)
async def analyze_unified_endpoint(
    request: Request,
    response: Response,
    # NOUVEAU: L'utilisateur doit choisir le type d'analyse
    analysis_type: AnalysisType = Form(
        ...,
        description="Le type d'analyse à effectuer: 'PLANT_PEST' ou 'SATELLITE_REMOTE_SENSING'."
    ),
    file: UploadFile = File(
        ...,
        description="Fichier image (JPEG,JPG, PNG). Doit être une photo de proximité ou une image satellite."
    )
):
    """
    **Rôle de ce service unifié :**
    
    1.  **Recevoir une image ET un type d'analyse.**
    2.  **Sélectionner le prompt expert** (Plante/Ravageur ou Télédétection).
    3.  **Appeler Gemini** avec le prompt et l'image appropriés.
    4.  **Gérer la rotation des clés API** en cas de panne.
    5.  **Découper les 'détections'** (lésions ou zones de stress) et les uploader sur Cloudinary.
    6.  **Retourner le JSON** structuré `AIAnalysisResponse`.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG, JPG ou PNG.")

    # --- 1. Sélection du Prompt ---
    if analysis_type == AnalysisType.PLANT_PEST:
        selected_prompt = PLANT_PEST_PROMPT
    elif analysis_type == AnalysisType.SATELLITE_REMOTE_SENSING:
        selected_prompt = SATELLITE_PROMPT
    else:
        # Normalement impossible grâce à l'Enum, mais c'est une sécurité
        raise HTTPException(status_code=400, detail="Type d'analyse inconnu.")

    image_bytes = await file.read()

    try:
        # --- 2. Appel de l'IA ---
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        
        gemini_response = await generate_gemini_analysis_with_key_rotation(
            prompt=selected_prompt,  # Passe le prompt sélectionné
            image_part=image_part,
            config=generation_config
        )

        analysis_data = json.loads(gemini_response.text)

        # --- 3. Logique de découpage (inchangée et réutilisable) ---
        # Cette logique fonctionne parfaitement pour les deux cas:
        # - Cas Plante: Découpe la lésion/ravageur.
        # - Cas Satellite: Découpe la zone de stress/déficience.
        if analysis_data.get("detections"):
            original_image = Image.open(io.BytesIO(image_bytes))
            width, height = original_image.size

            for detection in analysis_data["detections"]:
                if "boundingBox" not in detection:
                    continue # Ignore les détections sans BBox

                bbox = detection["boundingBox"]
                
                # Coordonnées pour le découpage
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
                    folder="pestai_detections_v9" # Nouveau dossier pour la v9
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


# --- Événements de Démarrage et Point de Santé (inchangés) ---
@app.on_event("startup")
async def startup():
    """Initialise le cache en mémoire au démarrage de l'application."""
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("Cache en mémoire initialisé. Service unifié v9.0.0 prêt.")

@app.get("/", include_in_schema=False)
def read_root():
    """Endpoint de santé pour vérifier que le service est en ligne."""
    return {"message": "PestAI - Unified Analysis Microservice v9.0.0 est opérationnel."}
