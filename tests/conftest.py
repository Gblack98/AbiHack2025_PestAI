import os

# Variables d'environnement factices pour les tests (avant tout import de l'app)
os.environ.setdefault("GEMINI_API_KEYS", "test_key_1,test_key_2")
os.environ.setdefault("CLOUDINARY_CLOUD_NAME", "test_cloud")
os.environ.setdefault("CLOUDINARY_API_KEY", "test_api_key")
os.environ.setdefault("CLOUDINARY_API_SECRET", "test_api_secret")
