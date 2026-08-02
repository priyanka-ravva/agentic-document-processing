"""Base interface shared by all workflow agents."""

from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.graph.state import AgentState


class BaseAgent(ABC):
    """Base class for all document workflow agents."""

    name: str = "base_agent"

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm

    @abstractmethod
    def invoke(self, state: AgentState) -> AgentState:
        """Run the agent and return an updated state."""
