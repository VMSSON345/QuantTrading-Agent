import time
import litellm
from loguru import logger
from ..utils.prompt_loader import load_config

# Doc config 1 lan khi import
_cfg = load_config("settings")
_llm_cfg = load_config("llm")


def _get_agent_cfg(agent_name: str) -> dict:
    """Lay config cho agent cu the tu llm.yaml."""
    return _llm_cfg.get(agent_name, _llm_cfg.get("default", {}))


class BaseAgent:
    def __init__(self, model: str = None, agent_name: str = None, api_key: str = None):
        # Uu tien: tham so truyen vao > llm.yaml > default
        if model:
            self.model = model
        elif agent_name:
            cfg = _get_agent_cfg(agent_name)
            self.model = cfg.get("model", "groq/llama-3.1-8b-instant")
        else:
            default = _llm_cfg.get("default", {})
            self.model = default.get("model", "groq/llama-3.1-8b-instant")

        self._agent_name = agent_name
        self.api_key = api_key  

    def _get_temperature(self, override: float = None) -> float:
        if override is not None:
            return override
        if self._agent_name:
            cfg = _get_agent_cfg(self._agent_name)
            return cfg.get("temperature", 0.7)
        return 0.7

    def _get_max_tokens(self, override: int = None) -> int:
        if override is not None:
            return override
        if self._agent_name:
            cfg = _get_agent_cfg(self._agent_name)
            return cfg.get("max_tokens", 1024)
        return 1024

    def _chat(self, messages: list, temperature: float = None,
              max_retries: int = 3, max_tokens: int = None) -> str:
        temp   = self._get_temperature(temperature)
        tokens = self._get_max_tokens(max_tokens)

        for attempt in range(max_retries):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                    api_key=self.api_key,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err:
                    wait = min(180, 30 * (2 ** attempt))
                    logger.warning(f"Rate limit 429 — cho {wait}s (lan {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM error: {e}")
                    raise
        raise RuntimeError("Vuot qua so lan retry do rate limit")
