import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import cloudinary.uploader
from PIL import Image

logger = logging.getLogger(__name__)


def _upload_crop(args: tuple) -> tuple[int, str | None]:
    """Upload une image croppée sur Cloudinary. Retourne (index, url_ou_None)."""
    idx, buffer, folder, class_name = args
    try:
        result = cloudinary.uploader.upload(buffer, folder=folder)
        return idx, result.get("secure_url")
    except Exception as e:
        logger.warning(f"Échec upload Cloudinary pour '{class_name}' : {e}")
        return idx, None


def crop_and_upload(image_bytes: bytes, detections: List[dict], folder: str) -> List[dict]:
    """
    Pour chaque détection, découpe la zone délimitée par la boundingBox
    et l'uploade sur Cloudinary en parallèle.

    En cas d'échec (image corrompue, coordonnées invalides, erreur Cloudinary),
    l'erreur est loguée et croppedImageUrl reste null — le reste de la réponse
    n'est pas affecté.

    Args:
        image_bytes: Contenu brut de l'image originale.
        detections:  Liste des détections renvoyées par Gemini.
        folder:      Dossier Cloudinary de destination.

    Returns:
        La liste des détections enrichies avec `croppedImageUrl`.
    """
    try:
        original = Image.open(io.BytesIO(image_bytes))
        width, height = original.size
    except Exception as e:
        logger.warning(f"Impossible d'ouvrir l'image pour le découpage : {e}")
        return detections

    # Phase 1 : découpage (rapide, CPU) — séquentiel
    upload_tasks = []
    for i, detection in enumerate(detections):
        detection.setdefault("croppedImageUrl", None)

        bbox = detection.get("boundingBox")
        if not bbox:
            continue

        try:
            coords = (
                int(bbox["x_min"] * width),
                int(bbox["y_min"] * height),
                int(bbox["x_max"] * width),
                int(bbox["y_max"] * height),
            )

            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                logger.warning(
                    f"BoundingBox invalide pour '{detection.get('className', '?')}' : {coords}"
                )
                continue

            cropped = original.crop(coords)
            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG")
            buffer.seek(0)
            upload_tasks.append((i, buffer, folder, detection.get("className", "?")))

        except Exception as e:
            logger.warning(
                f"Échec du découpage pour '{detection.get('className', '?')}' : {e}"
            )

    # Phase 2 : upload en parallèle (I/O réseau — ThreadPoolExecutor)
    if upload_tasks:
        max_workers = min(len(upload_tasks), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_upload_crop, task): task[0] for task in upload_tasks}
            for future in as_completed(futures):
                idx, url = future.result()
                if url:
                    detections[idx]["croppedImageUrl"] = url

    return detections
