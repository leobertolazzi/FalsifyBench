"""
Multi-Agent Game Engine

Implements the turn-based inductive reasoning game with Player and Oracle agents.
The paper models run through OpenAI, Azure OpenAI, or Together AI.
"""

import json
import traceback
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import os
import time

try:
    import openai
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AzureOpenAI = None
    OPENAI_AVAILABLE = False

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False


DEFAULT_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"


class ActionType(Enum):
    """Player action types."""
    TEST = "test"
    GUESS = "guess"


def _is_azure_gpt_5_2_chat_model(model: str) -> bool:
    """Return True only for the exact Azure-hosted GPT model name."""
    return isinstance(model, str) and model.strip().lower() == "gpt-5.2-chat"


def _detect_provider_for_model(model: str) -> str:
    """Infer the provider from the model name."""
    model_lower = model.lower()
    if model_lower.startswith("openai/gpt-oss") or model_lower.startswith("gpt-oss"):
        return "together"
    if _is_azure_gpt_5_2_chat_model(model):
        return "azure"
    if "gpt" in model_lower:
        return "openai"
    return "together"


def _create_provider_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the provider client and return resolved connection settings."""
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip() or None
        if not resolved_api_key:
            raise ValueError(
                "Missing API key for OpenAI provider. "
                "Set OPENAI_API_KEY in your environment (or .env)."
            )
        return {
            "api_key": resolved_api_key,
            "client": openai.OpenAI(api_key=resolved_api_key),
            "api_version": None,
            "azure_endpoint": None,
        }

    if provider == "azure":
        if not OPENAI_AVAILABLE or AzureOpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        resolved_api_key = (api_key or os.getenv("AZURE_OPENAI_API_KEY") or "").strip() or None
        azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip() or None
        api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_OPENAI_API_VERSION).strip()
        if not resolved_api_key or not azure_endpoint:
            raise ValueError(
                "Missing Azure OpenAI configuration for model "
                f"{model}. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                "in your environment (or .env)."
            )
        return {
            "api_key": resolved_api_key,
            "client": AzureOpenAI(
                api_key=resolved_api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            ),
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
        }

    if provider == "together":
        if not TOGETHER_AVAILABLE:
            raise ImportError("together package not installed. Run: pip install together")
        resolved_api_key = (api_key or os.getenv("TOGETHER_API_KEY") or "").strip() or None
        if not resolved_api_key:
            raise ValueError(
                "Missing API key for Together provider. "
                "Set TOGETHER_API_KEY in your environment (or .env)."
            )
        return {
            "api_key": resolved_api_key,
            "client": Together(api_key=resolved_api_key),
            "api_version": None,
            "azure_endpoint": None,
        }

    raise ValueError(f"Unsupported provider: {provider}")


def _is_api_timeout_error(error: Exception) -> bool:
    """Best-effort check for provider timeout errors."""
    name = type(error).__name__.lower()
    message = str(error).lower()
    combined = f"{name} {message}"
    return (
        "apitimeouterror" in combined
        or "request timed out" in combined
        or "timed out" in combined
        or "timeout" in combined
    )


@dataclass
class TestAction:
    """Player's test action."""
    action_type: Literal["test"] = "test"
    proposed_items: List[str] = None
    current_hypothesis: str = ""

    def __post_init__(self):
        if self.proposed_items is None:
            self.proposed_items = []


@dataclass
class GuessAction:
    """Player's guess action."""
    action_type: Literal["guess"] = "guess"
    guessed_property: str = ""


@dataclass
class OracleResponse:
    """Oracle's response to player actions."""
    conforms: Optional[bool] = None  # For test actions
    correct: Optional[bool] = None   # For guess actions
    feedback: str = ""


@dataclass
class HypothesisOracleResponse:
    """Internal oracle judgment about the player's hypothesis and intended test type."""
    conforms: Optional[bool] = None  # Whether items actually conform to the player's hypothesis
    intention: Optional[str] = None  # "confirm" or "falsify" inferred from hypothesis+reasoning
    feedback: str = ""


@dataclass
class GameTurn:
    """Record of a single game turn."""
    turn_number: int
    action: Dict[str, Any]
    oracle_response: Dict[str, Any]
    player_reasoning: str = ""
    inferred_intention: Optional[str] = None  # "confirm" or "falsify" inferred from hypothesis+items
    hypothesis_oracle_response: Optional[Dict[str, Any]] = None  # internal, not shown to player


class PlayerAgent:
    """
    LLM-based player agent that attempts to discover the target property.

    Supports the providers used in the paper:
    - Azure OpenAI: gpt-5.2-chat.
    - OpenAI: gpt-5-mini and gpt-5-nano.
    - Together: the remaining open and open-weight models.
    """

    def __init__(self, model: str = "gpt-5-nano-2025-08-07",
                 provider: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout_retry_attempts: int = 3,
                 timeout_retry_initial_delay: float = 1.5,
                 timeout_retry_backoff_factor: float = 2.0):
        self.model = model
        self.timeout_retry_attempts = max(0, int(timeout_retry_attempts))
        self.timeout_retry_initial_delay = max(0.0, float(timeout_retry_initial_delay))
        self.timeout_retry_backoff_factor = max(1.0, float(timeout_retry_backoff_factor))

        self.provider = provider or _detect_provider_for_model(model)
        client_config = _create_provider_client(self.provider, model, api_key=api_key)
        self.api_key = client_config["api_key"]
        self.client = client_config["client"]
        self.api_version = client_config["api_version"]
        self.azure_endpoint = client_config["azure_endpoint"]

        self.conversation_history = []

    def _call_with_timeout_retries(self, call_fn):
        """Retry only on timeout-like API errors with exponential backoff."""
        attempts = self.timeout_retry_attempts + 1
        delay = self.timeout_retry_initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return call_fn()
            except Exception as e:
                last_error = e
                if (not _is_api_timeout_error(e)) or attempt >= attempts:
                    raise
                if delay > 0:
                    time.sleep(delay)
                delay *= self.timeout_retry_backoff_factor
        if last_error is not None:
            raise last_error

    def initialize_game(self, initial_examples: List[str], max_turns: int = 20) -> str:
        """Initialize the game with initial examples."""
        system_prompt = """You are playing a property discovery game. You are given 3 items that share a common property.
Your goal is to guess the property by testing new items.

You have a maximum of {max_turns} turns.

What to expect from the oracle:
- After a TEST action, the oracle will reply with: "Oracle's judgment: CONFORM" or "Oracle's judgment: DO NOT CONFORM".
- After a GUESS action, the oracle will reply with: "Oracle's judgment: INCORRECT" if you're wrong; if you're correct the game ends.

In each turn, you can perform one of two actions:
1. Propose 3 new items to test your current hypothesis
2. Make a guess of the property if you think you know it

The purpose of testing is to gather information to refine your hypothesis. 
Once you believe you have enough information, you can make a guess.
If your guess is correct, you win. If it's incorrect, you can keep testing-guessing until you find the right property or run out of turns.

Your response must be valid JSON in one of these formats:

For testing:
{{
    "action": "test",
    "items": ["item1", "item2", "item3"],
    "hypothesis": "your current hypothesis",
    "reasoning": "explain your thinking"
}}

For guessing:
{{
    "action": "guess",
    "property": "your final guess as a single word or phrase",
    "reasoning": "explain why you're confident"
}}
""".format(max_turns=max_turns)

        initial_message = f"""Initial examples that share a common property:
1. {initial_examples[0]}
2. {initial_examples[1]}
3. {initial_examples[2]}

What is your first move? Respond with JSON only."""

        self.conversation_history = [
            {"role": "user", "content": initial_message}
        ]

        return system_prompt

    def take_turn(
        self,
        system_prompt: str,
        feedback: Optional[str] = None,
        low_reasoning_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Take a turn in the game.

        Args:
            system_prompt: System instructions
            feedback: Oracle's feedback from previous turn

        Returns:
            Parsed action dictionary
        """
        if feedback:
            self.conversation_history.append({
                "role": "user",
                "content": feedback
            })

        effective_system_prompt = system_prompt
        if low_reasoning_mode:
            effective_system_prompt = (
                system_prompt
                + "\n\nIMPORTANT: Keep reasoning minimal and concise. "
                + "Return strictly one JSON object only; no markdown and no extra text."
            )

        # Call appropriate API based on provider
        if self.provider in ("openai", "azure"):
            # OpenAI and Azure OpenAI use the same chat-completions format.
            messages = [{"role": "system", "content": effective_system_prompt}] + self.conversation_history
            base_kwargs = {
                "model": self.model,
                "max_completion_tokens": 10000,
                "messages": messages,
            }
            try:
                kwargs = dict(base_kwargs)
                if low_reasoning_mode:
                    kwargs["reasoning"] = {"effort": "low"}
                response = self._call_with_timeout_retries(
                    lambda: self.client.chat.completions.create(**kwargs)
                )
            except TypeError:
                fallback_kwargs = dict(base_kwargs)
                response = self._call_with_timeout_retries(
                    lambda: self.client.chat.completions.create(**fallback_kwargs)
                )
            assistant_message = response.choices[0].message.content
        elif self.provider == "together":
            messages = [{"role": "system", "content": effective_system_prompt}] + self.conversation_history
            kwargs = {
                "model": self.model,
                "max_tokens": 10000,
                "messages": messages,
            }
            if "deepseek" in self.model.lower():
                kwargs["reasoning"] = {"enabled": False}
            elif "kimi" in self.model.lower():
                kwargs["reasoning"] = {"enabled": not low_reasoning_mode}
                kwargs["temperature"] = 0.6 if low_reasoning_mode else 1.0
            else:
                pass
            response = self._call_with_timeout_retries(
                lambda: self.client.chat.completions.create(**kwargs)
            )
            assistant_message = response.choices[0].message.content

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        # Parse JSON response
        def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
            """Best-effort extraction of the first JSON object from a model response."""
            if not isinstance(text, str):
                return None

            cleaned = text.strip()

            # Handle fenced code blocks ```json ... ```
            if cleaned.startswith("```"):
                fence_end = cleaned.rfind("```")
                if fence_end > 0:
                    inner = cleaned.strip("`")
                    # If it was ```json\n... we want everything after the first newline
                    if "\n" in inner:
                        inner = inner.split("\n", 1)[1]
                    cleaned = inner.strip()

            # Direct parse
            try:
                parsed = json.loads(cleaned)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                pass

            # Scan for a decodable JSON object starting at any '{'
            decoder = json.JSONDecoder()
            for idx, ch in enumerate(cleaned):
                if ch != "{":
                    continue
                try:
                    obj, _end = decoder.raw_decode(cleaned[idx:])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    continue

            return None

        extracted = _extract_first_json_object(assistant_message)
        if extracted is None:
            preview = assistant_message
            if isinstance(preview, str) and len(preview) > 800:
                preview = preview[:800] + "..."
            raise ValueError(
                "Could not parse a JSON action from the model response. "
                "Expected a single JSON object. "
            )
        return extracted


class OracleAgent:
    """
    LLM-based oracle agent that judges whether items conform to the target property.

    Supports the providers used in the paper:
    - Azure OpenAI: gpt-5.2-chat.
    - OpenAI: gpt-5-mini and gpt-5-nano.
    - Together: the remaining open and open-weight models.
    """

    def __init__(self, target_property: str,
                 model: str = "gpt-5-nano-2025-08-07",
                 provider: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout_retry_attempts: int = 3,
                 timeout_retry_initial_delay: float = 1.5,
                 timeout_retry_backoff_factor: float = 2.0):
        self.target_property = target_property
        self.model = model
        self.timeout_retry_attempts = max(0, int(timeout_retry_attempts))
        self.timeout_retry_initial_delay = max(0.0, float(timeout_retry_initial_delay))
        self.timeout_retry_backoff_factor = max(1.0, float(timeout_retry_backoff_factor))

        self.provider = provider or _detect_provider_for_model(model)
        client_config = _create_provider_client(self.provider, model, api_key=api_key)
        self.api_key = client_config["api_key"]
        self.client = client_config["client"]
        self.api_version = client_config["api_version"]
        self.azure_endpoint = client_config["azure_endpoint"]

    def _call_with_timeout_retries(self, call_fn):
        """Retry only on timeout-like API errors with exponential backoff."""
        attempts = self.timeout_retry_attempts + 1
        delay = self.timeout_retry_initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return call_fn()
            except Exception as e:
                last_error = e
                if (not _is_api_timeout_error(e)) or attempt >= attempts:
                    raise
                if delay > 0:
                    time.sleep(delay)
                delay *= self.timeout_retry_backoff_factor
        if last_error is not None:
            raise last_error

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of the first JSON object from a model response."""
        if not isinstance(text, str):
            return None

        cleaned = text.strip()

        # Handle fenced code blocks ```json ... ```
        if cleaned.startswith("```"):
            fence_end = cleaned.rfind("```")
            if fence_end > 0:
                inner = cleaned.strip("`")
                if "\n" in inner:
                    inner = inner.split("\n", 1)[1]
                cleaned = inner.strip()

        # Direct parse
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        # Scan for a decodable JSON object starting at any '{'
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(cleaned):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(cleaned[idx:])
                return obj if isinstance(obj, dict) else None
            except Exception:
                continue

        return None

    @staticmethod
    def _coerce_bool_field(payload: Dict[str, Any], *candidate_keys: str) -> Optional[bool]:
        """Extract a boolean-like field from a parsed JSON payload."""
        for key in candidate_keys:
            if key not in payload:
                continue
            value = payload.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "yes", "y", "1", "correct", "conform", "conforms"}:
                    return True
                if normalized in {"false", "no", "n", "0", "incorrect", "not conform", "does not conform", "do not conform"}:
                    return False
        return None

    def _chat_create(
        self,
        *,
        prompt: str,
        max_completion_tokens: int,
        force_json: bool,
        reasoning_effort: Optional[str] = None,
    ) -> Any:
        """Wrapper to optionally request JSON-only responses, with graceful fallback.

        OpenAI and Azure use chat completions; Together uses its native SDK.
        Strict JSON controls are requested where supported, with a graceful fallback.
        """
        if self.provider == "together":
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_completion_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if "glm" in self.model.lower():
                kwargs["reasoning"] = {"enabled": False}
            elif "qwen" in self.model.lower():
                kwargs["reasoning"] = {"enabled": False}
                kwargs["temperature"] = 0.7
            elif "kimi" in self.model.lower():
                kwargs["reasoning"] = {"enabled": False}
                kwargs["temperature"] = 0.6
            return self._call_with_timeout_retries(
                lambda: self.client.chat.completions.create(**kwargs)
            )

        kwargs = {
            "model": self.model,
            "max_completion_tokens": max_completion_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        try:
            return self._call_with_timeout_retries(
                lambda: self.client.chat.completions.create(**kwargs)
            )
        except TypeError:
            # Older SDKs / endpoints might not support response_format/reasoning/max_completion_tokens
            fallback_kwargs = dict(kwargs)
            for key in ("response_format", "reasoning"):
                fallback_kwargs.pop(key, None)
            return self._call_with_timeout_retries(
                lambda: self.client.chat.completions.create(**fallback_kwargs)
            )

    def judge_items(self, items: List[str]) -> OracleResponse:
        """
        Judge whether the proposed items conform to the target property.

        Args:
            items: List of items to judge

        Returns:
            OracleResponse with judgment
        """
        if not isinstance(items, list) or len(items) != 3:
            raise ValueError(f"Oracle expected exactly 3 items to judge, got: {items!r}")

        prompt = f"""You are an oracle in a property discovery game. The target property is: "{self.target_property}"

The player has proposed these items:
1. {items[0]}
2. {items[1]}
3. {items[2]}

Do ALL three items conform to the property "{self.target_property}"?

Respond with ONLY a JSON object:
{{
    "conforms": true or false,
    "explanation": "brief explanation of your judgment"
}}"""

        if self.provider in ("openai", "azure", "together"):
            result_text = ""
            last_error: Optional[Exception] = None
            for attempt in range(2):
                try:
                    attempt_prompt = prompt
                    if attempt > 0:
                        attempt_prompt += "\n\nIMPORTANT: Return ONLY a JSON object (no prose, no markdown)."
                    
                    max_tokens = 10000
                    response = self._chat_create(
                        prompt=attempt_prompt,
                        max_completion_tokens=max_tokens,
                        force_json=True,
                        reasoning_effort="low",
                    )
                    # content can be None in some tool-call scenarios; normalize to ""
                    result_text = response.choices[0].message.content or ""
                    parsed = self._extract_first_json_object(result_text)
                    if parsed is None:
                        raise ValueError(
                            "Oracle returned non-JSON or empty response. "
                        )
                    conforms_value = self._coerce_bool_field(
                        parsed,
                        "conforms",
                        "conform",
                        "all_conform",
                        "all_conforms",
                        "is_conforming",
                    )
                    if conforms_value is None:
                        raise ValueError(
                            "Oracle JSON is missing a usable 'conforms' boolean field. "
                            f"Available keys: {sorted(parsed.keys())}. "
                        )
                    result = {
                        "conforms": conforms_value,
                        "explanation": parsed.get("explanation") or parsed.get("reasoning") or "",
                    }
                    break
                except Exception as e:
                    last_error = e
                    result = None
            if result is None:
                raise last_error  # type: ignore[misc]
        else:
            raise ValueError(f"Unsupported provider for oracle judgment: {self.provider}")

        return OracleResponse(
            conforms=result["conforms"],
            feedback=result.get("explanation", "")
        )

    def judge_items_against_property(
        self,
        property_hypothesis: str,
        items: List[str],
        player_reasoning: str = "",
    ) -> HypothesisOracleResponse:
        """Judge items vs the player's hypothesis and infer intended test type.

        We infer intention primarily from the player's stated hypothesis + reasoning (what they *expect*),
        while also judging whether the items actually conform to that hypothesis.
        """
        if not isinstance(items, list) or len(items) != 3:
            raise ValueError(f"Oracle expected exactly 3 items to judge, got: {items!r}")

        prompt = f"""You are an oracle in a property discovery game. The property to evaluate is: "{property_hypothesis}"

The player has proposed these items:
1. {items[0]}
2. {items[1]}
3. {items[2]}

Player's current hypothesis:
"{property_hypothesis}"

Player's reasoning for choosing these items:
"{player_reasoning}"

Tasks:
1) Infer the player's *intention* for this TEST based on their hypothesis + reasoning.
   - "confirm" = they expect all three items to conform to their hypothesis
   - "falsify" = they expect at least one item to NOT conform to their hypothesis
2) Separately, judge whether all three items actually conform to the hypothesis.

Respond with ONLY a JSON object:
{{
    "intention": "confirm" or "falsify",
    "expected_conforms": true or false,
    "conforms": true or false,
    "explanation": "brief explanation"
}}"""

        if self.provider in ("openai", "azure", "together"):
            result_text = ""
            last_error: Optional[Exception] = None
            for attempt in range(2):
                try:
                    attempt_prompt = prompt
                    if attempt > 0:
                        attempt_prompt += "\n\nIMPORTANT: Return ONLY a JSON object (no prose, no markdown)."
                    
                    max_tokens = 10000
                        
                    response = self._chat_create(
                        prompt=attempt_prompt,
                        max_completion_tokens=max_tokens,
                        force_json=True,
                        reasoning_effort="low",
                    )
                    result_text = response.choices[0].message.content or ""
                    parsed = self._extract_first_json_object(result_text)
                    if parsed is None:
                        raise ValueError(
                            "Oracle returned non-JSON or empty response. "
                        )
                    intention_value: Optional[str] = None
                    raw_intention = (
                        parsed.get("intention")
                        or parsed.get("intent")
                        or parsed.get("inferred_intention")
                    )
                    if isinstance(raw_intention, str):
                        normalized_intention = raw_intention.strip().lower()
                        if normalized_intention in {"confirm", "confirmatory", "confirmation"}:
                            intention_value = "confirm"
                        elif normalized_intention in {"falsify", "falsification", "disconfirm", "disconfirmatory"}:
                            intention_value = "falsify"

                    expected_value = self._coerce_bool_field(
                        parsed,
                        "expected_conforms",
                        "expects_conforms",
                        "expected",
                        "should_conform",
                    )
                    if intention_value is None and isinstance(expected_value, bool):
                        intention_value = "confirm" if expected_value else "falsify"

                    conforms_value = self._coerce_bool_field(
                        parsed,
                        "conforms",
                        "conform",
                        "all_conform",
                        "all_conforms",
                        "is_conforming",
                    )

                    if intention_value is None and conforms_value is None:
                        raise ValueError(
                            "Oracle JSON is missing usable intention/conforms fields. "
                            f"Available keys: {sorted(parsed.keys())}. "
                        )
                    result = {
                        "intention": intention_value,
                        "conforms": conforms_value,
                        "explanation": parsed.get("explanation") or parsed.get("reasoning") or "",
                    }
                    break
                except Exception as e:
                    last_error = e
                    result = None
            if result is None:
                raise last_error  # type: ignore[misc]
        else:
            raise ValueError(f"Unsupported provider for oracle judgment: {self.provider}")

        intention = result.get("intention")
        if isinstance(intention, str):
            intention = intention.lower().strip()
        else:
            intention = None

        conforms = result.get("conforms")
        if not isinstance(conforms, bool):
            conforms = None

        return HypothesisOracleResponse(
            conforms=conforms,
            intention=intention,
            feedback=result.get("explanation", "")
        )

    def judge_guess(self, guessed_property: str) -> OracleResponse:
        """
        Judge whether the guessed property matches the target.

        Args:
            guessed_property: Player's guess

        Returns:
            OracleResponse with judgment
        """
        prompt = f"""You are an oracle in a property discovery game. The target property is: "{self.target_property}"

The player has guessed: "{guessed_property}"

Are these semantically equivalent (same meaning, possibly different wording)?

Respond with ONLY a JSON object:
{{
    "correct": true or false,
    "explanation": "brief explanation"
}}"""

        if self.provider in ("openai", "azure", "together"):
            result_text = ""
            last_error: Optional[Exception] = None
            for attempt in range(2):
                try:
                    attempt_prompt = prompt
                    if attempt > 0:
                        attempt_prompt += "\n\nIMPORTANT: Return ONLY a JSON object (no prose, no markdown)."
                    response = self._chat_create(
                        prompt=attempt_prompt,
                        max_completion_tokens=10000,
                        force_json=True,
                        reasoning_effort="low",
                    )
                    result_text = response.choices[0].message.content or ""
                    parsed = self._extract_first_json_object(result_text)
                    if parsed is None:
                        raise ValueError(
                            "Oracle returned non-JSON or empty response. "
                        )
                    result = parsed
                    break
                except Exception as e:
                    last_error = e
                    result = None
            if result is None:
                raise last_error  # type: ignore[misc]
        else:
            raise ValueError(f"Unsupported provider for oracle judgment: {self.provider}")

        return OracleResponse(
            correct=result["correct"],
            feedback=result.get("explanation", "")
        )


class GameEngine:
    """
    Main game engine that orchestrates the player-oracle interaction.
    """

    def __init__(self, game_instance: Dict[str, Any], max_turns: int = 20,
                 player_model: str = "gpt-5-nano-2025-08-07",
                 oracle_model: str = "gpt-5-nano-2025-08-07",
                 player_provider: Optional[str] = None,
                 oracle_provider: Optional[str] = None,
                 verbose: bool = False,
                 max_tests: Optional[int] = None,
                 max_action_retries: int = 2,
                 timeout_retry_attempts: int = 3,
                 timeout_retry_initial_delay: float = 1.5,
                 timeout_retry_backoff_factor: float = 2.0):
        self.game_instance = game_instance
        self.max_turns = max_turns
        self.verbose = verbose
        self.max_tests = max_tests
        self.max_action_retries = max(0, int(max_action_retries))
        self.target_property = game_instance["target_hypothesis"]
        self.initial_examples = game_instance["initial_examples"]

        resolved_player_model = player_model
        resolved_oracle_model = oracle_model
        resolved_player_provider = player_provider
        resolved_oracle_provider = oracle_provider

        self.player = PlayerAgent(
            model=resolved_player_model,
            provider=resolved_player_provider,
            timeout_retry_attempts=timeout_retry_attempts,
            timeout_retry_initial_delay=timeout_retry_initial_delay,
            timeout_retry_backoff_factor=timeout_retry_backoff_factor,
        )
        self.oracle = OracleAgent(target_property=self.target_property,
                                  model=resolved_oracle_model,
                                  provider=resolved_oracle_provider,
                                  timeout_retry_attempts=timeout_retry_attempts,
                                  timeout_retry_initial_delay=timeout_retry_initial_delay,
                                  timeout_retry_backoff_factor=timeout_retry_backoff_factor)

        self.turns: List[GameTurn] = []
        self.game_over = False
        self.success = False
        self.error: Optional[Dict[str, Any]] = None

    def _validate_player_action(self, action: Dict[str, Any]) -> Optional[str]:
        """Return None if valid; otherwise return a feedback string to send back to the player."""
        if not isinstance(action, dict):
            return (
                "Oracle: Invalid response format. Your response must be a single JSON object. "
                "Respond with JSON only."
            )

        action_type = action.get("action")
        if action_type not in {"test", "guess"}:
            return (
                "Oracle: Missing/invalid 'action'. Use 'test' or 'guess'. "
                "Respond with JSON only."
            )

        if action_type == "test":
            items = action.get("items")
            if not isinstance(items, list) or len(items) != 3 or not all(isinstance(x, str) and x.strip() for x in items):
                return (
                    "Oracle: Invalid TEST action. Provide exactly 3 non-empty strings in 'items'. "
                    "Respond with JSON only."
                )

            hypothesis = action.get("hypothesis", "")
            if hypothesis is None:
                # Normalize null to empty string
                action["hypothesis"] = ""
            elif not isinstance(hypothesis, str):
                return (
                    "Oracle: Invalid TEST action. 'hypothesis' must be a string. "
                    "Respond with JSON only."
                )

            reasoning = action.get("reasoning", "")
            if reasoning is None:
                action["reasoning"] = ""
            elif not isinstance(reasoning, str):
                return (
                    "Oracle: Invalid TEST action. 'reasoning' must be a string. "
                    "Respond with JSON only."
                )

        if action_type == "guess":
            prop = action.get("property")
            if not isinstance(prop, str) or not prop.strip():
                return (
                    "Oracle: Invalid GUESS action. Provide a non-empty string in 'property'. "
                    "Respond with JSON only."
                )

            reasoning = action.get("reasoning", "")
            if reasoning is None:
                action["reasoning"] = ""
            elif not isinstance(reasoning, str):
                return (
                    "Oracle: Invalid GUESS action. 'reasoning' must be a string. "
                    "Respond with JSON only."
                )

        return None

    def play_game(self) -> Dict[str, Any]:
        """
        Play a complete game and return results.

        Returns:
            Dictionary with game results and statistics
        """
        system_prompt = self.player.initialize_game(self.initial_examples, max_turns=self.max_turns)
        feedback = None
        test_actions_taken = 0

        for turn_num in range(1, self.max_turns + 1):
            if self.verbose:
                print("---")
            try:
                # Player takes action (with limited retries for malformed output)
                action: Optional[Dict[str, Any]] = None
                last_action_error: Optional[Exception] = None

                def _collect_player_action(low_reasoning_mode: bool) -> Optional[Dict[str, Any]]:
                    nonlocal feedback, last_action_error
                    candidate_action: Optional[Dict[str, Any]] = None
                    for attempt in range(self.max_action_retries + 1):
                        try:
                            candidate_action = self.player.take_turn(
                                system_prompt,
                                feedback,
                                low_reasoning_mode=low_reasoning_mode,
                            )
                            validation_feedback = self._validate_player_action(candidate_action)
                            if validation_feedback is None:
                                # Enforce hard cap on number of TEST actions.
                                # If exceeded, re-prompt the player to GUESS (within the same turn).
                                if (
                                    self.max_tests is not None
                                    and candidate_action.get("action") == "test"
                                    and test_actions_taken >= self.max_tests
                                ):
                                    feedback = (
                                        "Oracle: You reached the maximum number of TEST actions. "
                                        "You must now make a GUESS of the property. "
                                        "After your guess is judged, you may continue testing if needed. "
                                        "Respond with JSON only."
                                    )
                                    if self.verbose:
                                        print(
                                            f"Turn {turn_num} - TEST rejected (max_tests={self.max_tests}). "
                                            "Requesting GUESS instead."
                                        )
                                    candidate_action = None
                                    continue

                                return candidate_action

                            # Invalid action shape; ask the player to reformat.
                            if self.verbose:
                                print(f"Turn {turn_num} - Invalid action schema:", candidate_action)
                            feedback = validation_feedback
                            candidate_action = None
                        except Exception as e:
                            last_action_error = e
                            if self.verbose:
                                mode = "low-reasoning" if low_reasoning_mode else "normal"
                                print(
                                    f"Turn {turn_num} - Player output parse/API error ({mode}, attempt {attempt + 1}): {e}"
                                )
                            # Ask the player to output valid JSON only.
                            feedback = (
                                "Oracle: I could not parse your response as a single JSON object. "
                                "Respond with JSON only in the specified format."
                            )
                            candidate_action = None
                    return None

                action = _collect_player_action(low_reasoning_mode=False)

                if action is None:
                    # Retry once with reduced reasoning requirements.
                    feedback = (
                        "Oracle: Your last responses were invalid. "
                        "Retry with minimal reasoning and strict JSON only. "
                        "Keep fields short and output exactly one JSON object."
                    )
                    if self.verbose:
                        print(f"Turn {turn_num} - Retrying player action with low reasoning mode.")
                    action = _collect_player_action(low_reasoning_mode=True)

                if action is None:
                    # Exhausted retries in both normal and low-reasoning modes; fail with clear context.
                    raise RuntimeError(
                        f"Failed to obtain a valid player action after {2 * (self.max_action_retries + 1)} attempt(s)."
                    ) from last_action_error

                if self.verbose:
                    print(f"Turn {turn_num} - Player action:", action)
                # Oracle responds
                if action.get("action") == "test":
                    items = action.get("items", [])
                    hypothesis = action.get("hypothesis", "")

                    # Internal-only: infer whether this TEST is confirmatory or falsificatory
                    inferred_intention: Optional[str] = None
                    hypothesis_oracle_response: Optional[HypothesisOracleResponse] = None
                    try:
                        if isinstance(hypothesis, str) and hypothesis.strip():
                            hypothesis_oracle_response = self.oracle.judge_items_against_property(
                                hypothesis,
                                items,
                                player_reasoning=action.get("reasoning", "") or "",
                            )
                            # Prefer intention inferred from reasoning; fall back to conforms if missing.
                            if hypothesis_oracle_response.intention in {"confirm", "falsify"}:
                                inferred_intention = hypothesis_oracle_response.intention
                            elif isinstance(hypothesis_oracle_response.conforms, bool):
                                inferred_intention = "confirm" if hypothesis_oracle_response.conforms else "falsify"
                    except Exception:
                        # Don't break the game if this auxiliary inference fails
                        inferred_intention = None
                        hypothesis_oracle_response = None

                    if self.verbose:
                        if hypothesis_oracle_response is None:
                            print(f"Turn {turn_num} - Intention: {inferred_intention}")
                        else:
                            print(
                                f"Turn {turn_num} - Intention: {inferred_intention} "
                                f"(hypothesis_conforms={hypothesis_oracle_response.conforms})"
                            )

                    # Main oracle: judge against the true target property
                    oracle_response = self.oracle.judge_items(items)

                    test_actions_taken += 1

                    feedback = f"""Oracle's judgment: {"CONFORM" if oracle_response.conforms else "DO NOT CONFORM"}

What is your next move? Respond with JSON only."""
                    if self.verbose:
                        print(f"Turn {turn_num} - Oracle judgment:", "CONFORM" if oracle_response.conforms else "DO NOT CONFORM")

                    turn = GameTurn(
                        turn_number=turn_num,
                        action=action,
                        oracle_response=asdict(oracle_response),
                        player_reasoning=action.get("reasoning", ""),
                        inferred_intention=inferred_intention,
                        hypothesis_oracle_response=asdict(hypothesis_oracle_response) if hypothesis_oracle_response else None
                    )
                    self.turns.append(turn)

                elif action.get("action") == "guess":
                    guessed = action.get("property", "")
                    oracle_response = self.oracle.judge_guess(guessed)

                    if self.verbose:
                        print(
                            f"Turn {turn_num} - Oracle judgment:",
                            "CORRECT" if oracle_response.correct else "INCORRECT",
                        )

                    # Reset test budget after any guess is judged (correct or incorrect).
                    test_actions_taken = 0

                    turn = GameTurn(
                        turn_number=turn_num,
                        action=action,
                        oracle_response=asdict(oracle_response),
                        player_reasoning=action.get("reasoning", "")
                    )
                    self.turns.append(turn)

                    if oracle_response.correct:
                        self.success = True
                        self.game_over = True
                        break
                    else:
                        feedback = """Oracle's judgment: INCORRECT

The property you guessed is not correct. Continue testing. Respond with JSON only."""

            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error in turn {turn_num}: {type(e).__name__}: {e}")
                print(tb)

                # Record error details for downstream analysis
                self.error = {
                    "turn": turn_num,
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": tb,
                    "last_feedback": feedback,
                    "provider_player": getattr(self.player, "provider", None),
                    "provider_oracle": getattr(self.oracle, "provider", None),
                    "model_player": getattr(self.player, "model", None),
                    "model_oracle": getattr(self.oracle, "model", None),
                    "last_assistant_message": (
                        self.player.conversation_history[-1].get("content")
                        if getattr(self.player, "conversation_history", None)
                        and isinstance(self.player.conversation_history, list)
                        and self.player.conversation_history
                        and isinstance(self.player.conversation_history[-1], dict)
                        else None
                    ),
                }
                self.game_over = True
                break

        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Compile game results and statistics."""
        return {
            "game_id": self.game_instance["game_id"],
            "target_property": self.target_property,
            "sampling_hypothesis": self.game_instance["sampling_hypothesis"],
            "distance_group": self.game_instance["distance_group"],
            "distance": self.game_instance["distance"],
            "success": self.success,
            "turns_to_solution": len(self.turns) if self.success else None,
            "total_turns": len(self.turns),
            "max_turns_reached": len(self.turns) >= self.max_turns and not self.success,
            "turns": [asdict(turn) for turn in self.turns],
            "error": self.error,
        }
