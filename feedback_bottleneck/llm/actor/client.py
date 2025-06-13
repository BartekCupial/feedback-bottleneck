import base64
import logging
import os
import re
import time
from collections import namedtuple
from io import BytesIO
from typing import Optional

import httpx

from feedback_bottleneck.config.args import LLMClientArgs

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
    ],
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """Base class for LLM client wrappers.

    Provides common functionality for interacting with different LLM APIs, including
    handling retries and common configuration settings. Subclasses should implement
    the `generate` method specific to their LLM API.
    """

    def __init__(self, client_config):
        """Initialize the LLM client wrapper with configuration settings.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay
        self.alternate_roles = client_config.alternate_roles

    def generate(self, messages, sampling_params):
        """Generate a response from the LLM given a list of messages.

        This method should be overridden by subclasses.

        Args:
            messages (list): A list of messages to send to the LLM.
            sampling_params (dict): Dictionary containing sampling arguments.

        Returns:
            LLMResponse: The response from the LLM.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retries upon failure.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.

        Raises:
            Exception: If the function fails after the maximum number of retries.
        """
        retries = 0
        response = None
        last_exc: Optional[Exception] = None
        while retries < self.max_retries and response is None:
            try:
                response = func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                retries += 1
                logger.error(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}")
                sleep_time = self.delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)

        if response is not None:
            return response
        else:
            raise Exception(
                f"Failed to execute {func.__name__} after {self.max_retries} retries. Last error: {last_exc}"
            ) from last_exc


_CONTEXT_LEN_RE = re.compile(r"maximum context length is (\\d+) tokens", re.IGNORECASE)
_PROMPT_TOKENS_RE = re.compile(r"request has (\\d+) input tokens", re.IGNORECASE)


def process_image_openai(image):
    """Process an image for OpenAI API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for OpenAI.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for OpenAI
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_claude(image):
    """Process an image for Anthropic's Claude API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for Claude.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for Anthropic
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
    }


class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, client_config):
        """Initialize the OpenAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")

            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            elif self.client_name.lower() == "nvidia":
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA client")
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                # If base_url is provided, use it (OpenAI-compatible servers).
                if self.base_url and self.base_url.strip():
                    self.client = OpenAI(api_key=api_key, base_url=self.base_url)
                else:
                    self.client = OpenAI(api_key=api_key)
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the OpenAI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                new_content.append(process_image_openai(msg.attachment))
            if self.alternate_roles and converted_messages and converted_messages[-1]["role"] == msg.role:
                converted_messages[-1]["content"].extend(new_content)
            else:
                converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages, sampling_params):
        """Generate a response from the OpenAI API given a list of messages.

        Args:
            messages (list): A list of message objects.
            sampling_params (dict): Dictionary containing sampling arguments.

        Returns:
            LLMResponse: The response from the OpenAI API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        max_tokens = int(sampling_params.get("max_tokens", 1024))
        api_kwargs = {
            "messages": converted_messages,
            "model": self.model_id,
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
        }

        # Only enable logprobs for vLLM client
        if self.client_name.lower() == "vllm":
            logprobs_val = sampling_params.get("logprobs")
            if logprobs_val:
                api_kwargs["logprobs"] = True
                # If an integer is provided (e.g. 5), use it as top_logprobs
                if isinstance(logprobs_val, int) and logprobs_val > 0:
                    api_kwargs["top_logprobs"] = logprobs_val

        if self.client_name.lower() == "openai":
            token_param_name = "max_completion_tokens"
        else:
            token_param_name = "max_tokens"

        def api_call(current_max_tokens: int):
            api_kwargs[token_param_name] = current_max_tokens
            return self.client.chat.completions.create(**api_kwargs)

        retries = 0
        response = None
        last_exc: Optional[Exception] = None

        while retries < self.max_retries and response is None:
            try:
                response = api_call(max_tokens)
            except Exception as e:
                last_exc = e
                msg = str(e)

                max_len_match = _CONTEXT_LEN_RE.search(msg)
                prompt_match = _PROMPT_TOKENS_RE.search(msg)
                if max_len_match and prompt_match:
                    max_len = int(max_len_match.group(1))
                    prompt_tokens = int(prompt_match.group(1))
                    allowed = max_len - prompt_tokens
                    if allowed < 1:
                        allowed = 1
                    if max_tokens > allowed:
                        logger.warning(
                            "OpenAI request exceeded context window (max_len=%s prompt=%s). "
                            "Reducing max_tokens from %s to %s and retrying.",
                            max_len,
                            prompt_tokens,
                            max_tokens,
                            allowed,
                        )
                        max_tokens = allowed
                        retries += 1
                        continue

                retries += 1
                logger.error(
                    "Retryable error during api_call: %s. Retry %s/%s",
                    e,
                    retries,
                    self.max_retries,
                )
                sleep_time = self.delay * (2 ** (retries - 1))
                time.sleep(sleep_time)

        if response is None:
            raise Exception(
                f"Failed to execute api_call after {self.max_retries} retries. Last error: {last_exc}"
            ) from last_exc

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )


class GoogleGenerativeAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with Google's Generative AI API."""

    def __init__(self, client_config):
        """Initialize the GoogleGenerativeAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Generative AI client if not already initialized."""
        if not self._initialized:
            import google.generativeai as genai

            self.model = genai.GenerativeModel(self.model_id)
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Generative AI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the Generative AI API.
        """
        # Convert standard Message objects to Gemini's format
        converted_messages = []
        for msg in messages:
            parts = []
            role = msg.role
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
            if msg.content:
                parts.append(msg.content)
            if msg.attachment is not None:
                parts.append(msg.attachment)
            converted_messages.append(
                {
                    "role": role,
                    "parts": parts,
                }
            )
        return converted_messages

    def extract_completion(self, response):
        """Extract the completion text from the API response.

        Args:
            response: The response object from the API.

        Returns:
            str: The extracted completion text.

        Raises:
            Exception: If response is None or missing expected fields.
        """
        if not response:
            raise Exception("Response is None, cannot extract completion.")

        candidates = getattr(response, "candidates", [])
        if not candidates:
            raise Exception("No candidates found in the response.")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            raise Exception("No content found in the candidate.")

        content_parts = getattr(content, "parts", [])
        if not content_parts:
            raise Exception("No content parts found in the candidate.")

        text = getattr(content_parts[0], "text", None)
        if text is None:
            raise Exception("No text found in the content parts.")

        return text.strip()

    def generate(self, messages, sampling_params):
        """Generate a response from the Generative AI API given a list of messages.

        Args:
            messages (list): A list of message objects.
            sampling_params (dict): Dictionary containing sampling arguments.

        Returns:
            LLMResponse: The response from the Generative AI API.
        """
        self._initialize_client()
        import google.generativeai as genai

        converted_messages = self.convert_messages(messages)

        # Create kwargs dictionary for GenerationConfig from sampling_params
        gen_kwargs = {
            "max_output_tokens": sampling_params.get("max_tokens", 1024),
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
        }
        generation_config = genai.types.GenerationConfig(**gen_kwargs)

        def api_call():
            response = self.model.generate_content(
                converted_messages,
                generation_config=generation_config,
            )
            # Attempt to extract completion immediately after API call
            completion = self.extract_completion(response)
            # Return both response and completion if successful
            return response, completion

        try:
            # Execute the API call and extraction together with retries
            response, completion = self.execute_with_retries(api_call)

            # Check if the successful response contains an empty completion
            if not completion or completion.strip() == "":
                logger.warning(
                    f"Gemini returned an empty completion for model {self.model_id}. Returning default empty response."
                )
                return LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    stop_reason="empty_response",
                    input_tokens=(
                        getattr(response.usage_metadata, "prompt_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    output_tokens=(
                        getattr(response.usage_metadata, "candidates_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    reasoning=None,
                )
            else:
                # If completion is not empty, return the normal response
                return LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    stop_reason=(
                        getattr(response.candidates[0], "finish_reason", "unknown")
                        if response and getattr(response, "candidates", [])
                        else "unknown"
                    ),
                    input_tokens=(
                        getattr(response.usage_metadata, "prompt_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    output_tokens=(
                        getattr(response.usage_metadata, "candidates_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    reasoning=None,
                )
        except Exception as e:
            logger.error(f"API call failed after {self.max_retries} retries: {e}. Returning empty completion.")
            # Return a default response indicating failure
            return LLMResponse(
                model_id=self.model_id,
                completion="",
                stop_reason="error_max_retries",
                input_tokens=0,  # Assuming 0 tokens consumed if call failed
                output_tokens=0,
                reasoning=None,
            )


class ClaudeWrapper(LLMClientWrapper):
    """Wrapper for interacting with Anthropic's Claude API."""

    def __init__(self, client_config):
        """Initialize the ClaudeWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Claude client if not already initialized."""
        if not self._initialized:
            from anthropic import Anthropic

            self.client = Anthropic()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Claude API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the Claude API.
        """
        converted_messages = []
        for msg in messages:
            converted_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
            if converted_messages[-1]["role"] == "system":
                # Claude doesn't support system prompt and requires alternating roles
                converted_messages[-1]["role"] = "user"
                converted_messages.append({"role": "assistant", "content": "I'm ready!"})
            if msg.attachment is not None:
                converted_messages[-1]["content"].append(process_image_claude(msg.attachment))

        return converted_messages

    def generate(self, messages, sampling_params):
        """Generate a response from the Claude API given a list of messages.

        Args:
            messages (list): A list of message objects.
            sampling_params (dict): Dictionary containing sampling arguments.

        Returns:
            LLMResponse: The response from the Claude API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": sampling_params.get("max_tokens", 1024),
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
            }

            return self.client.messages.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.content[0].text.strip(),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
        )


class ReplicateWrapper(LLMClientWrapper):
    """Wrapper for interacting with the Replicate API."""

    def __init__(self, client_config):
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Replicate client with API token verification."""
        if not self._initialized:
            import replicate

            self.client = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"))
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to a single prompt for Replicate.

        Args:
            messages (list): A list of message objects.

        Returns:
            str: A string concatenating the roles and messages.
        """
        system_prompt = ""
        user_prompt = ""

        for msg in messages:
            if msg.role == "system":
                system_prompt += f"{msg.content}\n"
            else:
                user_prompt += f"{msg.role.upper()}: {msg.content}\n"

        return system_prompt.strip(), user_prompt.strip()

    def generate(self, messages, sampling_params):
        """Generate a response from the Replicate model given a list of messages.

        Args:
            messages (list): A list of message objects.
            sampling_params (dict): Dictionary containing sampling arguments.

        Returns:
            LLMResponse: The response from the Replicate model.
        """
        self._initialize_client()
        system_prompt, prompt = self.convert_messages(messages)

        def api_call():
            # Standard Replicate LLM input schema
            input_params = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": sampling_params.get("max_tokens", 1024),
                "temperature": sampling_params.get("temperature", 0.7),
                "top_p": sampling_params.get("top_p", 1.0),
            }

            # Replicate.run returns a generator for most text models
            output = self.client.run(self.model_id, input=input_params)
            return output

        response_gen = self.execute_with_retries(api_call)

        # Replicate returns an iterator of strings; join them to get the full response
        completion = "".join([str(item) for item in response_gen])

        return LLMResponse(
            model_id=self.model_id,
            completion=completion.strip(),
            stop_reason=None,  # Replicate may not give a specific finishing reason
            input_tokens=0,  # Usage tokens may not be directly available from Replicate
            output_tokens=0,
            reasoning=None,
        )


class OpenRouterWrapper(OpenAIWrapper):
    """Wrapper for interacting with the OpenRouter API."""

    def __init__(self, client_config):
        """Initialize the OpenRouterWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the OpenRouter client if not already initialized."""
        if not self._initialized:
            from openai import OpenAI

            self.client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
            self._initialized = True

    def generate(self, messages, sampling_params):
        """Generate a response from the OpenRouter API given a list of messages.

        Args:
            messages (list): A list of message objects.
            sampling_params (dict): Dictionary containing sampling arguments.

        Returns:
            LLMResponse: The response from the OpenRouter API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        # Extract sampling parameters
        max_tokens = int(sampling_params.get("max_tokens", 1024))
        temperature = sampling_params.get("temperature", 1.0)
        top_p = sampling_params.get("top_p", 1.0)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        # Handle cases where the response is None
        if response is None:
            logger.warning(f"OpenRouter returned None for model {self.model_id}. Returning default empty response.")
            return LLMResponse(
                model_id=self.model_id,
                completion="",
                stop_reason="none_response",
                input_tokens=0,
                output_tokens=0,
                reasoning=None,
            )

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )


def create_llm_client(client_config: LLMClientArgs):
    """
    Factory function to create the appropriate LLM client based on the client name.

    Args:
        client_config: Configuration object containing client-specific settings.

    Returns:
        callable: A factory function that returns an instance of the appropriate LLM client.
    """

    def client_factory():
        client_name_lower = client_config.client_name.lower()
        if "openrouter" in client_name_lower:
            return OpenRouterWrapper(client_config)
        elif "openai" in client_name_lower or "vllm" in client_name_lower or "nvidia" in client_name_lower:
            # NVIDIA uses OpenAI-compatible API, so we use the OpenAI wrapper
            return OpenAIWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_name_lower:
            return ClaudeWrapper(client_config)
        elif "replicate" in client_name_lower:
            return ReplicateWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
