import asyncio
import itertools
from typing import List


class KeyManager:
    """
    Gestionnaire de clés API asynchrone pour la rotation automatique
    en cas d'erreur de quota Gemini.
    Utilise un verrou asyncio pour être coroutine-safe.
    """

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("La liste des clés API ne peut pas être vide.")
        self.keys = keys
        self._key_iterator = itertools.cycle(keys)
        self._current_key = next(self._key_iterator)
        self._lock = asyncio.Lock()
        print(f"KeyManager initialisé avec {len(self.keys)} clé(s).")

    def get_current_key(self) -> str:
        return self._current_key

    async def rotate(self) -> str:
        async with self._lock:
            prev = self._current_key
            self._current_key = next(self._key_iterator)
            if prev != self._current_key:
                print(f"Quota dépassé → rotation vers clé ...{self._current_key[-4:]}")
            return self._current_key
