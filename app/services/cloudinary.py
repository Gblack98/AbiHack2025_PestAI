import io
import logging
from typing import List

import cloudinary.uploader
from PIL import Image

logger = logging.getLogger(__name__)


def crop_and_upload(image_bytes: bytes, detections: List[dict], folder: str) -> List[dict]:
    """
    Pour chaque détection, découpe la zone délimitée par la boundingBox
    et l'uploade sur Cloudinary.

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

    for detection in detections:
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

            # Vérification basique des coordonnées
            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                logger.warning(
                    f"BoundingBox invalide pour '{detection.get('className', '?')}' : {coords}"
                )
                continue

            cropped = original.crop(coords)
            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG")
            buffer.seek(0)

            result = cloudinary.uploader.upload(buffer, folder=folder)
            detection["croppedImageUrl"] = result.get("secure_url")

        except Exception as e:
            logger.warning(
                f"Échec du découpage/upload pour '{detection.get('className', '?')}' : {e}"
            )

    return detections
