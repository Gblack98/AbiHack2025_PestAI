import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise ValueError("Variable d'environnement GEMINI_API_KEYS non trouvée ou vide.")

# Clés dédiées aux endpoints voix (modèle gemini-2.0-flash, free tier généreux)
GEMINI_VOICE_KEYS = [k.strip() for k in os.getenv("GEMINI_VOICE_KEYS", "").split(",") if k.strip()]
if not GEMINI_VOICE_KEYS:
    GEMINI_VOICE_KEYS = GEMINI_API_KEYS  # fallback sur les clés principales

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY_ENV = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

GEMINI_MODEL = "gemini-3-flash-preview"
RATE_LIMIT = "15/minute"
CACHE_TTL = 86400  # 24h
