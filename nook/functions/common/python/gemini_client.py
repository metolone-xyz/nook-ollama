"""
LLM Client for Lambda functions.

This module provides a common interface for interacting with different LLM APIs including Gemini and Ollama.
"""

import os
from dataclasses import dataclass
from typing import Any, Union

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from google.genai.errors import ClientError, ServerError
import ollama
import subprocess
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeminiClientConfig:
    """Configuration for the Gemini client."""

    model: str = "gemini-2.0-flash-exp"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    response_mime_type: str = "text/plain"
    timeout: int = 60000
    use_search: bool = False

    def update(self, **kwargs) -> None:
        """Update the configuration with the given keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")


@dataclass
class OllamaClientConfig:
    """Configuration for the Ollama client."""

    model: str = "huggingface.co/rinna/qwen2.5-bakeneko-32b-instruct-gguf:latest"  # 日本語対応モデルをデフォルトに
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 8192
    host: str = "http://localhost:11434"
    
    def update(self, **kwargs) -> None:
        """Update the configuration with the given keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")


@dataclass
class PlamoClientConfig:
    """Configuration for the Plamo client."""

    model: str = "plamo-translate"  # 実際にはCLIツールを使用
    from_lang: str = "English"  # デフォルトは英語
    to_lang: str = "Japanese"  # デフォルトは日本語
    precision: str = "4bit"  # 4bit, 8bit, bf16
    
    def update(self, **kwargs) -> None:
        """Update the configuration with the given keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key: {key}")


class GeminiClient:
    """Client for interacting with the Gemini API."""

    def __init__(self, config: GeminiClientConfig | None = None, **kwargs):
        """
        Initialize the Gemini client.

        Parameters
        ----------
        config : GeminiClientConfig | None
            Configuration for the Gemini client.
            If not provided, default values will be used.
        """
        self._api_key = os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self._config = config or GeminiClientConfig()
        self._config.update(**kwargs)

        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=self._config.timeout),
        )
        self._chat = None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception(
            lambda e: isinstance(e, (ClientError, ServerError))
        ),
        before_sleep=lambda retry_state: logger.info(f"Retrying due to {retry_state.outcome.exception()}...")
    )
    def generate_content(
        self,
        contents: str | list[str],
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        """
        Generate content using the Gemini API.

        Parameters
        ----------
        contents : str | list[str]
            The content to generate from.
        system_instruction : str | None
            The system instruction to use.
        model : str | None
            The model to use.
            If not provided, the model from the config will be used.
        temperature : float | None
            The temperature to use.
            If not provided, the temperature from the config will be used.
        top_p : float | None
            The top_p to use.
            If not provided, the top_p from the config will be used.
        top_k : int | None
            The top_k to use.
            If not provided, the top_k from the config will be used.
        max_output_tokens : int | None
            The max_output_tokens to use.
            If not provided, the max_output_tokens from the config will be used.
        response_mime_type : str | None
            The response_mime_type to use.
            If not provided, the response_mime_type from the config will be used.

        Returns
        -------
        str
            The generated content.
        """
        if isinstance(contents, str):
            contents = [contents]

        config_params = {
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "top_k": top_k or self._config.top_k,
            "max_output_tokens": max_output_tokens or self._config.max_output_tokens,
            "response_mime_type": response_mime_type or self._config.response_mime_type,
            "safety_settings": self._get_default_safety_settings(),
        }

        if system_instruction:
            config_params["system_instruction"] = system_instruction

        response = self._client.models.generate_content(
            model=model or self._config.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_params),
        )

        return response.candidates[0].content.parts[0].text

    def create_chat(
        self,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        """
        Create a new chat session.

        Parameters
        ----------
        model : str | None
            The model to use.
            If not provided, the model from the config will be used.
        temperature : float | None
            The temperature to use.
            If not provided, the temperature from the config will be used.
        top_p : float | None
            The top_p to use.
            If not provided, the top_p from the config will be used.
        top_k : int | None
            The top_k to use.
            If not provided, the top_k from the config will be used.
        max_output_tokens : int | None
            The max_output_tokens to use.
            If not provided, the max_output_tokens from the config will be used.
        """
        config_params = {
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "top_k": top_k or self._config.top_k,
            "max_output_tokens": max_output_tokens or self._config.max_output_tokens,
            "response_modalities": ["TEXT"],
        }

        if self._config.use_search:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            config_params["tools"] = [google_search_tool]

        self._chat = self._client.chats.create(
            model=model or self._config.model,
            config=types.GenerateContentConfig(**config_params),
        )

    def send_message(self, message: str) -> str:
        """
        Send a message to the chat and get the response.

        Parameters
        ----------
        message : str
            The message to send.

        Returns
        -------
        str
            The response from the chat.

        Raises
        ------
        ValueError
            If no chat has been created.
        """
        if not self._chat:
            raise ValueError("No chat has been created. Call create_chat() first.")

        response = self._chat.send_message(message)
        return response.text

    def chat_with_search(self, message: str, model: str | None = None) -> str:
        """
        Create a new chat with search capability and send a message.

        This is a convenience method that combines create_chat() and send_message().

        Parameters
        ----------
        message : str
            The message to send.
        model : str | None
            The model to use.
            If not provided, the model from the config will be used.

        Returns
        -------
        str
            The response from the chat.
        """
        original_use_search = self._config.use_search
        self._config.use_search = True

        try:
            self.create_chat(model=model)
            return self.send_message(message)
        finally:
            self._config.use_search = original_use_search

    def _get_default_safety_settings(self) -> list[types.SafetySetting]:
        """
        Get the default safety settings.

        Returns
        -------
        list[types.SafetySetting]
            The default safety settings.
        """
        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]


def create_client(config: dict[str, Any] | None = None, **kwargs):
    """
    Create a client with the given configuration.
    
    The client type is determined by the LLM_PROVIDER environment variable.
    If not set, defaults to "ollama".

    Parameters
    ----------
    config : dict[str, Any] | None
        Configuration for the client.
        If not provided, default values will be used.

    Returns
    -------
    Union[GeminiClient, OllamaClient]
        The appropriate client based on the LLM_PROVIDER environment variable.
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    return create_unified_client(provider, config, **kwargs)


def create_gemini_client(config: dict[str, Any] | None = None, **kwargs) -> GeminiClient:
    """
    Create a Gemini client with the given configuration.

    Parameters
    ----------
    config : dict[str, Any] | None
        Configuration for the Gemini client.
        If not provided, default values will be used.

    Returns
    -------
    GeminiClient
        The Gemini client.
    """
    if config:
        client_config = GeminiClientConfig(
            model=config.get("model", "gemini-2.0-flash"),
            temperature=config.get("temperature", 1.0),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k", 40),
            max_output_tokens=config.get("max_output_tokens", 8192),
            response_mime_type=config.get("response_mime_type", "text/plain"),
            timeout=config.get("timeout", 60000),
            use_search=config.get("use_search", False),
        )
    else:
        client_config = None

    return GeminiClient(client_config, **kwargs)


class OllamaClient:
    """Client for interacting with Ollama."""

    def __init__(self, config: OllamaClientConfig | None = None, **kwargs):
        """
        Initialize the Ollama client.

        Parameters
        ----------
        config : OllamaClientConfig | None
            Configuration for the Ollama client.
            If not provided, default values will be used.
        """
        self._config = config or OllamaClientConfig()
        self._config.update(**kwargs)
        
        # Initialize Ollama client
        self._client = ollama.Client(host=self._config.host)

    def generate_content(
        self,
        contents: str | list[str],
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> str:
        """
        Generate content using Ollama.

        Parameters
        ----------
        contents : str | list[str]
            The content to generate from.
        system_instruction : str | None
            The system instruction to use.
        model : str | None
            The model to use.
            If not provided, the model from the config will be used.
        temperature : float | None
            The temperature to use.
            If not provided, the temperature from the config will be used.
        top_p : float | None
            The top_p to use.
            If not provided, the top_p from the config will be used.
        max_tokens : int | None
            The max_tokens to use.
            If not provided, the max_tokens from the config will be used.

        Returns
        -------
        str
            The generated content.
        """
        if isinstance(contents, list):
            contents = "\n".join(contents)

        # Prepare messages
        messages = []
        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction
            })
        
        messages.append({
            "role": "user", 
            "content": contents
        })

        # Prepare options
        options = {
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "num_predict": max_tokens or self._config.max_tokens,
        }

        try:
            response = self._client.chat(
                model=model or self._config.model,
                messages=messages,
                options=options
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating content with Ollama: {e}")
            raise

    def create_chat(self, **kwargs):
        """Create a new chat session. (Not implemented for Ollama)"""
        raise NotImplementedError("Chat sessions are not implemented for Ollama client.")

    def send_message(self, message: str) -> str:
        """Send a message to the chat. (Not implemented for Ollama)"""
        raise NotImplementedError("Chat sessions are not implemented for Ollama client.")

    def chat_with_search(self, message: str, model: str | None = None) -> str:
        """Chat with search capability. (Not implemented for Ollama)"""
        raise NotImplementedError("Search capability is not implemented for Ollama client.")


def create_ollama_client(config: dict[str, Any] | None = None, **kwargs) -> OllamaClient:
    """
    Create an Ollama client with the given configuration.

    Parameters
    ----------
    config : dict[str, Any] | None
        Configuration for the Ollama client.
        If not provided, default values will be used.

    Returns
    -------
    OllamaClient
        The Ollama client.
    """
    if config:
        client_config = OllamaClientConfig(
            model=config.get("model", "bakeneko-2025-3-17:latest"),
            temperature=config.get("temperature", 0.8),
            top_p=config.get("top_p", 0.9),
            max_tokens=config.get("max_tokens", 8192),
            host=config.get("host", "http://localhost:11434"),
        )
    else:
        client_config = None

    return OllamaClient(client_config, **kwargs)


class PlamoClient:
    """Client for interacting with Plamo translation model."""

    def __init__(self, config: PlamoClientConfig | None = None, **kwargs):
        """
        Initialize the Plamo client.

        Parameters
        ----------
        config : PlamoClientConfig | None
            Configuration for the Plamo client.
            If not provided, default values will be used.
        """
        self._config = config or PlamoClientConfig()
        self._config.update(**kwargs)

    def generate_content(
        self,
        contents: str | list[str],
        system_instruction: str | None = None,
        model: str | None = None,
        **kwargs
    ) -> str:
        """
        Generate content using Plamo translation.

        For translation tasks, this method will attempt to translate the content.
        For general text generation, it will process the content through translation.

        Parameters
        ----------
        contents : str | list[str]
            The content to translate or process.
        system_instruction : str | None
            Instructions for the translation (e.g., "Translate to English").
            If provided, will attempt to parse target language.
        model : str | None
            Not used for Plamo (always uses plamo-translate).
        **kwargs
            Additional arguments (not used for Plamo).

        Returns
        -------
        str
            The translated or processed content.
        """
        if isinstance(contents, list):
            contents = "\n".join(contents)

        # Parse target language from system instruction if provided
        to_lang = self._config.to_lang
        from_lang = self._config.from_lang
        
        if system_instruction:
            # Simple parsing for common translation instructions
            instruction_lower = system_instruction.lower()
            if "english" in instruction_lower or "英語" in instruction_lower:
                to_lang = "English"
            elif "japanese" in instruction_lower or "日本語" in instruction_lower:
                to_lang = "Japanese"
            elif "chinese" in instruction_lower or "中国語" in instruction_lower:
                to_lang = "Chinese"
            elif "korean" in instruction_lower or "韓国語" in instruction_lower:
                to_lang = "Korean"

        try:
            # Use plamo-translate CLI via subprocess
            cmd = [
                "plamo-translate",
                "--from", from_lang,
                "--to", to_lang,
                "--precision", self._config.precision,
                "--input", contents
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60秒のタイムアウト
            )
            
            if result.returncode != 0:
                logger.error(f"Plamo translation failed: {result.stderr}")
                raise RuntimeError(f"Plamo translation failed: {result.stderr}")
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            logger.error("Plamo translation timed out")
            raise RuntimeError("Plamo translation timed out")
        except Exception as e:
            logger.error(f"Error running plamo-translate: {e}")
            raise

    def create_chat(self, **kwargs):
        """Create a new chat session. (Not implemented for Plamo)"""
        raise NotImplementedError("Chat sessions are not implemented for Plamo client.")

    def send_message(self, message: str) -> str:
        """Send a message to the chat. (Not implemented for Plamo)"""
        raise NotImplementedError("Chat sessions are not implemented for Plamo client.")

    def chat_with_search(self, message: str, model: str | None = None) -> str:
        """Chat with search capability. (Not implemented for Plamo)"""
        raise NotImplementedError("Search capability is not implemented for Plamo client.")


def create_plamo_client(config: dict[str, Any] | None = None, **kwargs) -> PlamoClient:
    """
    Create a Plamo client with the given configuration.

    Parameters
    ----------
    config : dict[str, Any] | None
        Configuration for the Plamo client.
        If not provided, default values will be used.

    Returns
    -------
    PlamoClient
        The Plamo client.
    """
    if config:
        client_config = PlamoClientConfig(
            model=config.get("model", "plamo-translate"),
            from_lang=config.get("from_lang", "English"),
            to_lang=config.get("to_lang", "Japanese"),
            precision=config.get("precision", "4bit"),
        )
    else:
        client_config = None

    return PlamoClient(client_config, **kwargs)


def create_unified_client(provider: str = "ollama", config: dict[str, Any] | None = None, **kwargs):
    """
    Create a unified client that can switch between different LLM providers.

    Parameters
    ----------
    provider : str
        The provider to use ("gemini", "ollama", or "plamo").
    config : dict[str, Any] | None
        Configuration for the client.
        If not provided, default values will be used.

    Returns
    -------
    Union[GeminiClient, OllamaClient, PlamoClient]
        The appropriate client based on the provider.
    """
    if provider.lower() == "gemini":
        return create_gemini_client(config, **kwargs)
    elif provider.lower() == "ollama":
        return create_ollama_client(config, **kwargs)
    elif provider.lower() == "plamo":
        return create_plamo_client(config, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported providers are 'gemini', 'ollama', and 'plamo'.")
