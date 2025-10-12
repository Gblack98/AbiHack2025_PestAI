# =================================================================================
# ÉTAPE 1: "BUILDER" - Installation des dépendances dans un environnement isolé
# =================================================================================
# On utilise une image de base slim et on la nomme "builder"
FROM python:3.12-slim as builder

# Variables d'environnement pour optimiser Python et pip
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Création et activation d'un environnement virtuel dans /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copie du fichier de dépendances
WORKDIR /app
COPY requirements.txt .

# Mise à jour de pip et installation des dépendances
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt


# =================================================================================
# ÉTAPE 2: "FINAL" - Création de l'image de production légère et sécurisée
# =================================================================================
# On repart d'une image de base neuve pour la propreté et la légèreté
FROM python:3.12-slim

# Création d'un utilisateur non-root pour la sécurité (C'EST LE BON ENDROIT)
RUN addgroup --system app && adduser --system --ingroup app app

# On copie uniquement l'environnement virtuel avec les dépendances déjà installées
# depuis l'étape "builder".
COPY --from=builder /opt/venv /opt/venv

# On définit le répertoire de travail
WORKDIR /app

# On copie le code de notre application
COPY . .

# L'utilisateur 'app' devient propriétaire des fichiers
RUN chown -R app:app /app

# On bascule vers l'utilisateur non-root
USER app

# On expose le port 8000
EXPOSE 8000

# On s'assure que le shell utilisera les exécutables du venv
ENV PATH="/opt/venv/bin:$PATH"

# Commande pour lancer l'application avec Gunicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]
