import json
from urllib.parse import urlparse
from typing import AsyncGenerator, Dict, Optional
import httpx
from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, THINKING_ENABLED, THINKING_PARAM_MODE


def _infer_provider(base_url: str) -> str:
    """Guess the provider name from the base URL for display purposes."""
    host = urlparse(base_url).hostname or ""
    if "openai.com" in host:
        return "openai"
    if "deepseek.com" in host:
        return "deepseek"
    if "opencode.ai" in host:
        return "opencode"
    if "groq.com" in host:
        return "groq"
    if "localhost" in host or "127.0.0.1" in host:
        return "local"
    if "together" in host:
        return "together"
    if "anthropic" in host:
        return "anthropic"
    return "openai-compatible"


class LLMClient:
    """Generic OpenAI-compatible LLM client — works with DeepSeek, OpenAI, Groq, local models, etc."""

    def __init__(self):
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL
        self.provider = _infer_provider(self.base_url)
        self.client = httpx.AsyncClient(timeout=180.0)

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_thinking_param(self) -> dict:
        """Return the thinking mode parameter for the request body.

        The `thinking` field is DeepSeek/OpenCode-specific — OpenAI, Groq,
        and Ollama reject unknown top-level params. Gated by THINKING_PARAM_MODE:
        auto (default) sends it only for DeepSeek-family providers.
        """
        if THINKING_PARAM_MODE == "off":
            return {}
        if THINKING_PARAM_MODE == "auto" and self.provider not in ("deepseek", "opencode"):
            return {}
        mode = "enabled" if THINKING_ENABLED else "disabled"
        return {"thinking": {"type": mode}}

    async def generate(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        history: Optional[list] = None,
    ) -> Dict[str, str]:
        """Generate a response from the LLM. Returns {'content': ..., 'reasoning': ...}.

        ``history`` is an optional list of prior {role: 'user'|'assistant', content} turns,
        sent as preceding messages so follow-ups resolve against earlier context.
        """
        messages = [{"role": "user", "content": prompt}]
        if history:
            messages = [{"role": m["role"], "content": m["content"]} for m in history] + messages
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self._get_thinking_param(),
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()
            msg = data["choices"][0]["message"]
            return {
                "content": msg.get("content", ""),
                "reasoning": msg.get("reasoning_content", ""),
            }
        except httpx.HTTPError as e:
            raise RuntimeError(f"LLM API error ({self.provider}): {str(e)}")

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        history: Optional[list] = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream a response from the LLM. Yields {'type': 'reasoning'|'content', 'chunk': str}."""
        messages = [{"role": "user", "content": prompt}]
        if history:
            messages = [{"role": m["role"], "content": m["content"]} for m in history] + messages
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **self._get_thinking_param(),
        }

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._get_headers(),
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        delta = chunk["choices"][0]["delta"]
                        # Reasoning content (DeepSeek-style thinking tokens)
                        rc = delta.get("reasoning_content")
                        if rc:
                            yield {"type": "reasoning", "chunk": rc}
                        # Visible content
                        c = delta.get("content")
                        if c:
                            yield {"type": "content", "chunk": c}
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def check_health(self) -> bool:
        """Check if the LLM API is reachable and the key is valid."""
        # Local providers (Ollama, LM Studio) need no API key.
        if not self.api_key and self.provider != "local":
            return False
        try:
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10.0,
            )
            return response.status_code == 200
        except Exception:
            return False
