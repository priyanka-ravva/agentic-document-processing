"""Streamlit application for the Document Agent."""

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure the root project directory is in the sys path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.graph.state import create_initial_state
from src.graph.workflow import build_workflow
from src.utils.logging import configure_logging
from src.utils.trace_writer import save_run_trace


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="Agentic Document Extraction", page_icon="📄", layout="wide")

    # Load environment variables
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_settings()
    configure_logging(settings.log_level)

    st.title("📄 Agentic Document Extraction")
    st.markdown("Upload a PDF or Image (Invoice, Medical Record, Contract) to extract structured data automatically.")

    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document... this may take a moment."):
                # Save the uploaded file to a temporary file
                upload_stem = Path(uploaded_file.name).stem or "document"
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, prefix=f"{upload_stem}_", suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    # Build and run the workflow
                    app = build_workflow()
                    initial_state = create_initial_state(file_path=tmp_path)
                    
                    final_state = app.invoke(initial_state)
                    trace_path = save_run_trace(final_state)
                    
                    # Display the results
                    st.success("Document processed successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Extraction Summary")
                        st.write(f"**Detected Type:** {final_state.get('document_type', 'unknown').capitalize()}")
                        st.write(f"**Tool Used:** {final_state.get('selected_tool', 'unknown')}")
                        st.write(f"**Validation Score:** {final_state.get('validation_result', {}).get('quality_score', 'N/A')}")
                        st.write(f"**Trace Saved:** `{trace_path}`")
                        
                        missing_fields = final_state.get('validation_result', {}).get('missing_fields', [])
                        if missing_fields:
                            st.warning(f"**Missing Fields:** {', '.join(missing_fields)}")
                        
                        st.write(f"**Planner Reasoning:** {final_state.get('planner_reasoning', 'None')}")

                    with col2:
                        st.subheader("Structured JSON Data")
                        st.json(final_state.get("structured_output", {}))
                        
                    with st.expander("View Agent Trace Logs"):
                        st.json(final_state.get("logs", []))

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Cleanup the temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
