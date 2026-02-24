import logging

import google.api_core.exceptions
import google.generativeai as genai
from fastapi import HTTPException

from app.config import GEMINI_MODEL
from app.key_manager import KeyManager

logger = logging.getLogger(__name__)


async def call_gemini(
    prompt: str,
    image_part: dict | None,
    config: genai.types.GenerationConfig,
    key_manager: KeyManager,
) -> str:
    """
    Appelle l'API Gemini avec rotation automatique des clés en cas de quota dépassé.
    Retourne le texte brut de la réponse (JSON string).

    Raises:
        HTTPException 429 : toutes les clés ont atteint leur quota.
        HTTPException 503 : échec après rotation complète.
    """
    initial_key = key_manager.get_current_key()

    for _ in range(len(key_manager.keys)):
        current_key = key_manager.get_current_key()
        try:
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            contents = [prompt, image_part] if image_part else [prompt]
            response = await model.generate_content_async(
                contents,
                generation_config=config,
                request_options={"timeout": 120},
            )
            return response.text

        except (
            google.api_core.exceptions.ResourceExhausted,
            google.api_core.exceptions.PermissionDenied,
        ) as e:
            logger.warning(f"Quota/permission error sur clé ...{current_key[-4:]} : {e}")
            await key_manager.rotate()
            if key_manager.get_current_key() == initial_key:
                logger.error("Toutes les clés API Gemini ont dépassé leur quota.")
                raise HTTPException(
                    status_code=429,
                    detail="Toutes les clés API Gemini sont indisponibles ou ont dépassé leur quota.",
                )

    raise HTTPException(
        status_code=503,
        detail="Échec de l'analyse IA après rotation de toutes les clés disponibles.",
    )
