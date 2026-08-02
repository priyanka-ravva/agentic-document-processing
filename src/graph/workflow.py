"""LangGraph workflow assembly."""

from langgraph.graph import END, START, StateGraph

from src.agents.extractor import ExtractionAgent
from src.agents.planner import PlannerAgent
from src.agents.classifier import ClassifierAgent
from src.agents.qa import QAAgent
from src.agents.reflector import ReflectionAgent
from src.graph.router import route_reflector, route_selected_tool
from src.graph.state import AgentState, add_log
from src.tools.document_analyzer import analyze_document
from src.tools.ocr import extract_text_with_ocr
from src.tools.pdf_parser import extract_text_from_pdf


def document_analyzer_node(state: AgentState) -> AgentState:
    """Analyze the input document for planner routing."""

    metadata = analyze_document(state["file_path"])
    updated_state = state.copy()
    updated_state["document_metadata"] = metadata
    return add_log(
        updated_state,
        agent="document_analyzer",
        message="Document metadata collected.",
        metadata=metadata,
    )


def pdf_parser_node(state: AgentState) -> AgentState:
    """Run the PDF parser tool."""

    extracted_text = extract_text_from_pdf(state["file_path"])
    updated_state = state.copy()
    updated_state["extracted_text"] = extracted_text
    return add_log(
        updated_state,
        agent="pdf_parser_tool",
        message="PDF parser extracted embedded text.",
        character_count=len(extracted_text),
    )


def ocr_node(state: AgentState) -> AgentState:
    """Run the OCR tool."""

    extracted_text = extract_text_with_ocr(state["file_path"])
    updated_state = state.copy()
    updated_state["extracted_text"] = extracted_text
    return add_log(
        updated_state,
        agent="ocr_tool",
        message="OCR extracted text from rendered document pages.",
        character_count=len(extracted_text),
    )


from src.agents.vision_extractor import VisionExtractionAgent

def build_workflow():
    """Build and compile the LangGraph workflow."""

    workflow = StateGraph(AgentState)

    workflow.add_node("document_analyzer", document_analyzer_node)
    workflow.add_node("planner", PlannerAgent().invoke)
    workflow.add_node("pdf_parser", pdf_parser_node)
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("classifier", ClassifierAgent().invoke)
    workflow.add_node("extractor", ExtractionAgent().invoke)
    workflow.add_node("vision_extractor", VisionExtractionAgent().invoke)
    workflow.add_node("qa", QAAgent().invoke)
    workflow.add_node("reflector", ReflectionAgent().invoke)

    workflow.add_edge(START, "document_analyzer")
    workflow.add_edge("document_analyzer", "planner")
    workflow.add_conditional_edges(
        "planner",
        route_selected_tool,
        {
            "pdf_parser": "pdf_parser",
            "ocr": "ocr",
        },
    )
    workflow.add_edge("pdf_parser", "classifier")
    workflow.add_edge("ocr", "classifier")
    workflow.add_edge("classifier", "extractor")
    workflow.add_edge("extractor", "qa")
    workflow.add_edge("vision_extractor", "qa")
    workflow.add_edge("qa", "reflector")
    workflow.add_conditional_edges(
        "reflector",
        route_reflector,
        {
            "end": END,
            "retry": "extractor",
            "vision_retry": "vision_extractor",
        },
    )

    return workflow.compile()
