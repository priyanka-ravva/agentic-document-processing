from src.agents.planner import PlannerAgent
from src.graph.state import create_initial_state
from src.schemas.planner import ExtractionTool


def test_planner_selects_text_parser_for_supported_text_extensions() -> None:
    state = create_initial_state(file_path='sample_docs/test.txt')
    state['document_metadata'] = {
        'file_extension': '.txt',
        'text_length': 0,
        'has_embedded_text': False,
    }

    result = PlannerAgent().invoke(state)

    assert result['selected_tool'] == ExtractionTool.TEXT_PARSER
    assert 'TEXT_PARSER' in result['logs'][-1]['metadata']['selected_tool']
