"""Factory for retrieving agent prompts."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from src.prompts.classifier import get_classifier_system_prompt
from src.prompts.extractor import get_extractor_system_prompt
from src.prompts.extractors.invoice import get_invoice_extractor_system_prompt
from src.prompts.extractors.medical import get_medical_extractor_system_prompt
from src.prompts.extractors.contract import get_contract_extractor_system_prompt
from src.prompts.planner import get_planner_system_prompt
from src.prompts.qa import get_qa_system_prompt
from src.prompts.reflector import get_reflector_system_prompt


@dataclass(frozen=True)
class Prompt:
    """Container for prompt text and optional metadata."""

    name: str
    system_prompt: str
    context: Optional[Mapping[str, Any]] = None


class PromptFactory:
    """Central registry for agent prompts."""

    _PROMPTS: dict[str, Callable[[], str]] = {
        "planner": get_planner_system_prompt,
        "classifier": get_classifier_system_prompt,
        "extractor": get_extractor_system_prompt,
        "extractor_invoice": get_invoice_extractor_system_prompt,
        "extractor_medical": get_medical_extractor_system_prompt,
        "extractor_contract": get_contract_extractor_system_prompt,
        "qa": get_qa_system_prompt,
        "reflector": get_reflector_system_prompt,
    }

    @classmethod
    def get_prompt(
        cls,
        agent: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Prompt:
        """Return a prompt for the requested agent."""

        prompt_builder = cls._PROMPTS.get(agent)
        if prompt_builder is None:
            supported_agents = ", ".join(sorted(cls._PROMPTS))
            raise ValueError(f"Unknown prompt agent '{agent}'. Supported agents: {supported_agents}")

        return Prompt(
            name=agent,
            system_prompt=prompt_builder(),
            context=context,
        )
