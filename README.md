# 🌿 PestAI — Agricultural AI Analysis API

API de détection agronomique par intelligence artificielle, développée dans le cadre du hackathon **AbiHack 2025**.  
Elle analyse des images de plantes, de satellites et de drones pour identifier maladies, ravageurs et anomalies agricoles, avec des recommandations adaptées — optimisées pour une lecture vocale en **Wolof**.

---

## 🎯 Objectif

Fournir un microservice d'analyse d'images agricoles accessible via API REST, capable de :
- Détecter des maladies et ravageurs sur des cultures
- Analyser des parcelles depuis des images satellites ou drones
- Retourner des recommandations structurées (biologiques, chimiques, culturales)
- Générer des réponses simples adaptées à un système TTS (lecture vocale en Wolof)

---

## 🗂️ Structure du projet

```
AbiHack2025_PestAI/
│
├── main.py                          # API v11.0 — Version unifiée (production)
├── main2.py                         # API v8.5 — Version de référence (prompt universel)
├── Dockerfile                       # Build multi-étapes, sécurisé
├── .dockerignore                    # Fichiers exclus du build Docker
├── .gitignore                       # Fichiers exclus de Git
├── requirements.txt                 # Dépendances épinglées
├── .env.example                     # Template des variables d'environnement
└── Documentation_API_Hackathon.pdf  # Documentation officielle du hackathon
```

---

## 🔄 Versions de l'API

| Version | Fichier | Endpoint | Description |
|---|---|---|---|
| **v11.0** | `main.py` | `/api/v11/analyze-unified` | 3 types d'analyse spécialisés (Plante, Satellite, Drone) |
| **v8.5** | `main2.py` | `/api/v8/analyze-image` | Prompt universel, analyse plante/ravageur uniquement |

---

## 🦠 Types d'analyse (v11)

| Type | Description |
|---|---|
| `PLANT_PEST` | Photo en proximité — feuilles, tiges, insectes |
| `SATELLITE_REMOTE_SENSING` | Images satellites (Sentinel-2, Landsat) de parcelles |
| `DRONE_ANALYSIS` | Orthophotos haute résolution de zones agricoles |

---

## 📦 Technologies

| Outil | Rôle |
|---|---|
| FastAPI | Framework API REST asynchrone |
| Google Gemini (`gemini-2.5-flash`) | Modèle IA multimodal pour l'analyse d'images |
| Cloudinary | Stockage des zones détectées (crops) |
| Pydantic | Validation et sérialisation des données |
| slowapi | Rate limiting (15 req/min) |
| fastapi-cache2 | Cache en mémoire (TTL 24h, clé = hash SHA-256 de l'image) |
| Docker | Conteneurisation pour le déploiement |
| Gunicorn + Uvicorn | Serveur ASGI de production |

---

## ⚙️ Installation locale

### 1. Cloner le repo

```bash
git clone https://github.com/Gblack98/AbiHack2025_PestAI.git
cd AbiHack2025_PestAI
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env avec tes clés
```

### 5. Lancer l'API

```bash
# Version v11 (recommandée)
uvicorn main:app --reload --port 8000

# Version v8.5
uvicorn main2:app --reload --port 8001
```

L'API est disponible sur `http://localhost:8000`  
La documentation interactive Swagger est sur `http://localhost:8000/docs`

---

## 🐳 Déploiement Docker

```bash
# Build
docker build -t pestai-api .

# Run
docker run -p 8000:8000 --env-file .env pestai-api
```

---

## 🔑 Variables d'environnement

| Variable | Description |
|---|---|
| `GEMINI_API_KEYS` | Clés API Gemini séparées par des virgules (`clé1,clé2,clé3`) |
| `CLOUDINARY_CLOUD_NAME` | Nom du cloud Cloudinary |
| `CLOUDINARY_API_KEY` | Clé API Cloudinary |
| `CLOUDINARY_API_SECRET` | Secret API Cloudinary |

> Le système de **rotation automatique des clés** (`KeyManager`) bascule vers la clé suivante en cas de quota dépassé.

---

## 📡 Exemples d'utilisation

### Analyse d'une plante (v11)

```bash
curl -X POST "http://localhost:8000/api/v11/analyze-unified" \
  -F "analysis_type=PLANT_PEST" \
  -F "file=@feuille_malade.jpg"
```

### Analyse satellite (v11)

```bash
curl -X POST "http://localhost:8000/api/v11/analyze-unified" \
  -F "analysis_type=SATELLITE_REMOTE_SENSING" \
  -F "file=@parcelle_sentinel.png"
```

### Format de réponse

```json
{
  "subject": {
    "subjectType": "PLANT",
    "description": "Plant de Maïs (Zea mays)",
    "confidence": 0.97
  },
  "detections": [
    {
      "className": "Rouille commune du maïs",
      "confidenceScore": 0.89,
      "severity": "HIGH",
      "boundingBox": { "x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.7 },
      "croppedImageUrl": "https://res.cloudinary.com/...",
      "details": {
        "description": "Cette maladie couvre les feuilles de taches orange.",
        "impact": "Le rendement va baisser si rien n'est fait.",
        "recommendations": {
          "biological": [...],
          "chemical": [...],
          "cultural": [...]
        },
        "knowledgeBaseTags": ["rouille", "maïs", "champignon"]
      }
    }
  ]
}
```

---

## 👤 Auteur

**Gblack98** — [github.com/Gblack98](https://github.com/Gblack98)  
Hackathon AbiHack 2025
