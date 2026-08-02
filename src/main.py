"""CLI entry point for the Agentic Document Agent."""

import argparse
import json

from src.agents.extractor import ExtractionAgent
from src.agents.classifier import ClassifierAgent
from src.agents.planner import PlannerAgent
from src.agents.qa import QAAgent
from src.agents.reflector import ReflectionAgent
from src.config import get_settings
from src.graph.workflow import build_workflow
from src.graph.state import create_initial_state
from src.prompts.prompt_factory import PromptFactory
from src.utils.logging import configure_logging, get_logger
from src.utils.trace_writer import save_run_trace


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the Agentic Document Agent.")
    parser.add_argument(
        "--file",
        dest="file_path",
        help="Path to a PDF or image document to process.",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print final structured extraction JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a health check or process a document."""

    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    state = create_initial_state(file_path=args.file_path or "sample_docs/invoice_simple.pdf")
    prompt = PromptFactory.get_prompt(agent="planner", context=state)

    agents = [
        PlannerAgent(),
        ClassifierAgent(),
        ExtractionAgent(),
        QAAgent(),
        ReflectionAgent(),
    ]

    logger.info("Agentic Document Agent skeleton initialized.")
    logger.info("Configured Groq model: %s", settings.groq_model)
    logger.info("Loaded prompt: %s", prompt.name)
    logger.info("Registered agents: %s", ", ".join(agent.name for agent in agents))
    logger.info("Initial state keys: %s", ", ".join(state.keys()))

    if not args.file_path:
        logger.info("No --file provided. Health check complete.")
        logger.info("Run with: python -m src.main --file sample_docs/invoice_simple.pdf")
        return

    app = build_workflow()
    final_state = app.invoke(state)
    trace_path = save_run_trace(final_state)

    logger.info("Workflow complete.")
    logger.info("Selected tool: %s", final_state["selected_tool"])
    logger.info("Planner reason: %s", final_state["planner_reasoning"])
    logger.info("Extracted text characters: %s", len(final_state["extracted_text"]))
    logger.info("Detected document type: %s", final_state.get("document_type", "unknown"))
    logger.info("Validation score: %s", final_state["validation_result"].get("quality_score"))
    logger.info("Missing fields: %s", final_state["validation_result"].get("missing_fields", []))
    logger.info("State log events: %s", len(final_state["logs"]))
    logger.info("Trace saved: %s", trace_path)

    if args.show_json:
        print(json.dumps(final_state["structured_output"], indent=2))


if __name__ == "__main__":
    main()
