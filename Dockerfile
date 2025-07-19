FROM python:3.12-slim

# --- Étape 2: Configuration de l'Environnement ---
# On définit le répertoire de travail à l'intérieur du conteneur.
# Toutes les commandes suivantes s'exécuteront depuis ce dossier.
WORKDIR /app

# On définit des variables d'environnement pour optimiser le comportement de Python.
# 1. Empêche Python de bufferiser les sorties (stdout/stderr), ce qui est mieux pour le logging.
# 2. Empêche Python de créer des fichiers .pyc, inutiles dans un conteneur.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# --- Étape 3: Installation des Dépendances ---
# On copie UNIQUEMENT le fichier requirements.txt d'abord.
# Docker met en cache cette couche. Si le fichier ne change pas, il n'exécutera pas
# la longue étape d'installation à chaque fois, ce qui accélère les builds.
COPY requirements.txt .

# On exécute la commande pip pour installer toutes les dépendances listées.
# --no-cache-dir réduit la taille finale de l'image.
RUN pip install --default-timeout=500 --no-cache-dir -r requirements.txt
# --- Étape 4: Copie du Code de l'Application ---
# Maintenant que les dépendances sont installées, on copie le reste du code
# (votre fichier main.py et le .env pour la construction, si nécessaire).
COPY . .

# --- Étape 5: Exposition du Port ---
# On informe Docker que notre application écoutera sur le port 8000
# à l'intérieur du conteneur. C'est ce port que Gunicorn va utiliser.
EXPOSE 8000

# main:app: Indique à Gunicorn de lancer l'objet "app" qui se trouve dans le fichier "main.py".
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]
