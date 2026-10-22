"""The Gemini client that c2_correction calls (client.models.generate_content)."""
from ..config import settings


class LLMUnavailable(RuntimeError):
    pass


_client = None


def get_client():
    """Returns a google-genai client or raises LLMUnavailable (the pipeline then returns FALLBACK_RAW)."""
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise LLMUnavailable("GEMINI_API_KEY is not set")
        try:
            from google import genai
            _client = genai.Client(api_key=settings.GEMINI_API_KEY,
                                   http_options={"timeout": settings.LLM_TIMEOUT_S * 1000})
        except Exception as exc:
            raise LLMUnavailable(str(exc)) from exc
    return _client


def set_client(c) -> None:
    """Used by tests and evaluation scripts to plug in a fake or another client."""
    global _client
    _client = c
