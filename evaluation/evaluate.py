"""Evaluate the document agent against configured scenarios."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.state import create_initial_state
from src.graph.workflow import build_workflow
from src.utils.logging import configure_logging, get_logger
from src.utils.trace_writer import save_run_trace


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the Agentic Document Agent.")
    parser.add_argument(
        "--scenarios",
        default=str(Path(__file__).with_name("scenarios.json")),
        help="Path to the evaluation scenarios JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("evaluation_results.json")),
        help="Path where evaluation results JSON should be written.",
    )
    return parser.parse_args()


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load evaluation scenarios from JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def get_nested_value(payload: dict[str, Any], dotted_path: str) -> Any:
    """Read a nested value using dot notation."""

    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def evaluate_scenario(app, scenario: dict[str, Any]) -> dict[str, Any]:
    """Run and score one scenario."""

    start = time.perf_counter()
    final_state = app.invoke(create_initial_state(file_path=scenario["file_path"]))
    duration_seconds = round(time.perf_counter() - start, 2)
    trace_path = save_run_trace(final_state)

    structured_output = final_state.get("structured_output", {})
    missing_required_fields = [
        field
        for field in scenario.get("required_fields", [])
        if not get_nested_value(structured_output, field)
    ]

    document_type_passed = (
        final_state.get("document_type") == scenario.get("expected_document_type")
    )
    expected_tool = scenario.get("expected_tool")
    expected_tools = expected_tool if isinstance(expected_tool, list) else [expected_tool]
    tool_passed = final_state.get("selected_tool") in expected_tools
    required_fields_passed = len(missing_required_fields) == 0
    passed = document_type_passed and tool_passed and required_fields_passed

    return {
        "id": scenario["id"],
        "name": scenario["name"],
        "passed": passed,
        "duration_seconds": duration_seconds,
        "expected_document_type": scenario.get("expected_document_type"),
        "actual_document_type": final_state.get("document_type"),
        "expected_tool": scenario.get("expected_tool"),
        "actual_tool": final_state.get("selected_tool"),
        "missing_required_fields": missing_required_fields,
        "quality_score": final_state.get("validation_result", {}).get("quality_score"),
        "trace_path": str(trace_path),
    }


def main() -> None:
    """Run all evaluation scenarios."""

    args = parse_args()
    configure_logging()
    logger = get_logger(__name__)
    scenarios_path = Path(args.scenarios)
    scenarios = load_scenarios(scenarios_path)
    app = build_workflow()

    results = [evaluate_scenario(app, scenario) for scenario in scenarios]
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed_count = sum(1 for result in results if result["passed"])
    logger.info("Evaluation complete: %s/%s passed", passed_count, len(results))
    logger.info("Results saved: %s", output_path)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        logger.info(
            "%s | %s | type=%s | tool=%s | score=%s",
            status,
            result["id"],
            result["actual_document_type"],
            result["actual_tool"],
            result["quality_score"],
        )


if __name__ == "__main__":
    main()
