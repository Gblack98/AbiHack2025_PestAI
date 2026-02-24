# 🌿 PestAI — Agricultural AI Analysis API

API de détection agronomique par intelligence artificielle, développée dans le cadre du hackathon **AbiHack 2025**.  
Elle analyse des images de plantes, de satellites et de drones pour identifier maladies, ravageurs et anomalies agricoles — avec des recommandations optimisées pour une lecture vocale en **Wolof**.

---

## 🎯 Objectif

Fournir un microservice d'analyse d'images agricoles via API REST, capable de :
- Détecter maladies et ravageurs sur des cultures (proximité)
- Analyser des parcelles depuis des images satellites ou drones
- Retourner des recommandations structurées (biologiques, chimiques, culturales)
- Générer des réponses simples adaptées à un système TTS en Wolof

---

## 🗂️ Structure du projet

```
AbiHack2025_PestAI/
│
├── main.py                          # API v12 — Version production (entry point)
├── main2.py                         # API v8.5 — Version de référence (prompt universel)
│
├── app/                             # Modules de l'application
│   ├── config.py                    # Variables d'environnement et constantes
│   ├── models.py                    # Schémas Pydantic (requêtes / réponses)
│   ├── prompts.py                   # Prompts experts (Plante, Satellite, Drone)
│   ├── key_manager.py               # Rotation async des clés API Gemini
│   └── services/
│       ├── gemini.py                # Appel Gemini avec gestion d'erreurs
│       └── cloudinary.py            # Découpage des zones et upload Cloudinary
│
├── tests/
│   ├── conftest.py                  # Fixtures et env vars de test
│   └── test_api.py                  # Tests des endpoints et des modules
│
├── Dockerfile                       # Build multi-étapes, user non-root
├── .dockerignore
├── .gitignore
├── .env.example                     # Template des variables d'environnement
├── requirements.txt                 # Dépendances épinglées
└── Documentation_API_Hackathon.pdf  # Doc officielle du hackathon
```

---

## 🔄 Versions

| Version | Fichier | Endpoint | Description |
|---|---|---|---|
| **v12** | `main.py` | `/api/v12/analyze` | Architecture modulaire, 3 prompts spécialisés |
| **v8.5** | `main2.py` | `/api/v8/analyze-image` | Version de référence, prompt universel |

---

## 🦠 Types d'analyse (v12)

| Type | Description |
|---|---|
| `PLANT_PEST` | Photo en proximité — feuilles, tiges, insectes |
| `SATELLITE_REMOTE_SENSING` | Images satellites (Sentinel-2, Landsat) de parcelles |
| `DRONE_ANALYSIS` | Orthophotos haute résolution |

---

## 📦 Stack technique

| Outil | Rôle |
|---|---|
| FastAPI | Framework API REST asynchrone |
| Gemini 3 Flash Preview | Modèle IA multimodal (vision + texte) |
| Cloudinary | Stockage des zones détectées (crops) |
| Pydantic | Validation et sérialisation JSON |
| slowapi | Rate limiting — 15 req/min |
| fastapi-cache2 | Cache en mémoire (TTL 24h, clé = SHA-256 image) |
| Docker | Conteneurisation |
| Gunicorn + Uvicorn | Serveur ASGI de production |
| pytest | Tests unitaires et d'intégration |

---

## ⚙️ Installation locale

### 1. Cloner le repo

```bash
git clone https://github.com/Gblack98/AbiHack2025_PestAI.git
cd AbiHack2025_PestAI
```

### 2. Environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
```

### 3. Dépendances

```bash
pip install -r requirements.txt
```

### 4. Variables d'environnement

```bash
cp .env.example .env
# Remplir les valeurs dans .env
```

### 5. Lancer l'API

```bash
uvicorn main:app --reload --port 8000
```

Documentation interactive disponible sur `http://localhost:8000/docs`

---

## 🐳 Docker

```bash
docker build -t pestai-api .
docker run -p 8000:8000 --env-file .env pestai-api
```

---

## 🔑 Variables d'environnement

| Variable | Description |
|---|---|
| `GEMINI_API_KEYS` | Clés Gemini séparées par des virgules (`clé1,clé2,clé3`) |
| `CLOUDINARY_CLOUD_NAME` | Nom du cloud Cloudinary |
| `CLOUDINARY_API_KEY` | Clé API Cloudinary |
| `CLOUDINARY_API_SECRET` | Secret Cloudinary |

> Le `KeyManager` tourne automatiquement vers la clé suivante si le quota est dépassé.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Les tests mockent Gemini et Cloudinary — aucune clé réelle nécessaire.

---

## 📡 Exemple d'utilisation

```bash
curl -X POST "http://localhost:8000/api/v12/analyze" \
  -F "analysis_type=PLANT_PEST" \
  -F "file=@feuille.jpg"
```

**Réponse :**

```json
{
  "subject": {
    "subjectType": "PLANT",
    "description": "Plant de Maïs (Zea mays)",
    "confidence": 0.96
  },
  "detections": [
    {
      "className": "Rouille commune du maïs",
      "confidenceScore": 0.89,
      "severity": "HIGH",
      "boundingBox": { "x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.7 },
      "croppedImageUrl": "https://res.cloudinary.com/...",
      "details": {
        "description": "Taches orange sur les feuilles.",
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
