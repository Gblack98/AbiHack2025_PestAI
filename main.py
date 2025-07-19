import os
import json
from enum import Enum
from typing import List, Optional
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Configuration Initiale ---
# Charge les variables d'environnement (essentiel pour Docker et le développement local)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Vérification critique au démarrage du service
if not GEMINI_API_KEY:
    # Cette erreur arrêtera le lancement du conteneur si la clé n'est pas fournie, ce qui est une bonne pratique.
    raise ValueError("Clé API Gemini non trouvée. Veuillez la définir dans les variables d'environnement (ex: via un fichier .env).")

genai.configure(api_key=GEMINI_API_KEY)


# --- Structures de Données Pydantic (Le "Contrat" de données avec le backend) ---
# Ces modèles garantissent que la sortie de l'API est toujours structurée correctement.

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

# La réponse finale validée par FastAPI avant d'être envoyée au client.
class AIAnalysisResponse(BaseModel):
    subject: AnalysisSubject
    detections: List[Detection]


# --- Le "Golden Prompt" v8.1 (Le cerveau de l'IA) ---

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

# --- Initialisation de l'Application FastAPI ---

app = FastAPI(
    title="PestAI - IA Analysis Microservice",
    description="API v8.1. Un service spécialisé qui reçoit une image et retourne une analyse agronomique experte au format JSON. Destiné à être appelé par un backend principal (ex: Nest.js).",
    version="8.1.0"
)

# --- Le Point d'Entrée (Endpoint) du Microservice ---

@app.post(
    "/api/v8/analyze-image",
    response_model=AIAnalysisResponse,
    summary="Prend une image et retourne une analyse IA complète.",
    tags=["IA Analysis Service"]
)
async def analyze_image_endpoint(
    file: UploadFile = File(..., description="Fichier image (JPEG, PNG) de la plante ou du ravageur."),
):
    """
    **Rôle de ce service :**
    - Ne gère ni les utilisateurs, ni les cultures, ni la base de données.
    - Se concentre à 100% sur l'exécution de l'analyse IA la plus performante possible.
    - Retourne un JSON pur et validé, prêt à être consommé par le backend principal.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Format d'image non supporté. Utilisez JPEG ou PNG.")

    try:
        image_bytes = await file.read()
        
        # Préparation de l'entrée pour le modèle Gemini
        image_part = {"mime_type": file.content_type, "data": image_bytes}
        
        # Configuration pour forcer une sortie JSON, ce qui est plus robuste.
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        
        # Initialisation du modèle
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Appel asynchrone à l'API avec la configuration JSON
        response = await model.generate_content_async(
            [UNIVERSAL_PROMPT, image_part],
            generation_config=generation_config
        )

        # La réponse de l'IA est maintenant directement un JSON valide.
        # Pas besoin de nettoyer les "```json"
        analysis_json = json.loads(response.text)

        # FastAPI valide automatiquement que `analysis_json` correspond au `response_model`
        # avant de l'envoyer. Si la validation échoue, il lèvera une erreur 500.
        return analysis_json

    except json.JSONDecodeError:
        # Cette erreur est maintenant très improbable, mais reste une bonne sécurité.
        print(f"Erreur de parsing JSON critique. L'API n'a pas retourné un JSON valide.\nTexte reçu: {response.text}")
        raise HTTPException(status_code=502, detail="Réponse invalide du service d'IA (non-JSON).")
    
    except Exception as e:
        # Capture toutes les autres erreurs possibles (problèmes réseau, API Google, etc.)
        print(f"Erreur inattendue durant l'appel à l'API Gemini: {e}")
        raise HTTPException(status_code=503, detail=f"Erreur interne du service IA: {str(e)}")

@app.get("/", include_in_schema=False)
def read_root():
    """Endpoint de santé pour vérifier que le service est en ligne."""
    return {"message": "PestAI - IA Analysis Microservice v8.1 est opérationnel."}