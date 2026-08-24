"""Lightweight interview demo UI for the HackerRank Support Agent."""

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from support_agent.db import DocumentChunkRepository
from support_agent.embeddings import GeminiEmbeddingProvider
from support_agent.llm import OpenRouterProvider
from support_agent.models import Ticket
from support_agent.orchestrator import TriageOrchestrator
from support_agent.retrieval import SemanticRetriever

load_dotenv()

st.set_page_config(page_title="HackerRank Support Agent", page_icon="💬")

st.title("HackerRank Support Agent")
st.caption("Policy + retrieval + LLM support triage")

# Resolved relative to this file (not the working directory) so the Samples
# tab still finds the CSVs regardless of where `streamlit run` is invoked
# from.
SAMPLES_DIR = Path(__file__).resolve().parent / "public" / "samples"


@st.cache_resource
def get_agent():
    retriever = SemanticRetriever(
        embedding_provider=GeminiEmbeddingProvider.from_env(),
        repository=DocumentChunkRepository.from_env(),
        top_k=5,
        candidate_k=30,
        similarity_threshold=0.0,
        max_chunks_per_source=2,
    )
    return TriageOrchestrator(
        retriever,
        OpenRouterProvider.from_env(),
        min_retrieved_chunks=1,
    )

def process(issue, subject=""):
    if not issue.strip():
        return None
    return get_agent().process_ticket(
        Ticket(issue=issue.strip(), subject=subject.strip())
    )


@st.cache_data
def _load_sample_csv(path_str: str, _mtime: float) -> pd.DataFrame:
    # `_mtime` isn't used in the body, but including it in the cache key
    # means the cache invalidates itself if the CSV on disk changes,
    # without needing a manual "clear cache" during the demo.
    return pd.read_csv(path_str, encoding="utf-8-sig")


def render_sample_csv(filename: str, description: str):
    """Show one sample CSV as a table, with a download button, or a clear
    error if it isn't where it's expected to be."""

    path = SAMPLES_DIR / filename
    st.caption(description)

    if not path.exists():
        st.error(f"`{filename}` not found at `{path}`.")
        return

    try:
        df = _load_sample_csv(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not read `{filename}`: {type(exc).__name__}: {exc}")
        return

    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        data=path.read_bytes(),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )


quick, detailed, samples = st.tabs(["Quick Reply", "Detailed", "Samples"])

with quick:
    st.subheader("Customer-facing response")
    issue = st.text_area(
        "Support query",
        height=140,
        placeholder="Example: How long do HackerRank tests stay active?",
        key="quick_issue",
    )
    if st.button("Get Response", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                result = process(issue)
                if result:
                    if result.status == "replied":
                        st.success(result.response)
                    else:
                        st.warning(result.response)
                else:
                    st.warning("Enter a support query first.")
            except Exception as exc:
                st.error(f"Unable to process request: {type(exc).__name__}")

with detailed:
    st.subheader("Full agent result")
    subject = st.text_input(
        "Subject (optional)",
        placeholder="Example: Test expiration",
        key="detail_subject",
    )
    issue = st.text_area(
        "Support query",
        height=140,
        placeholder="Enter a support request.",
        key="detail_issue",
    )
    if st.button("Analyze Ticket", type="primary", use_container_width=True):
        with st.spinner("Running triage..."):
            try:
                result = process(issue, subject)
                if result:
                    st.markdown("### Response")
                    if result.status == "replied":
                        st.success(result.response)
                    else:
                        st.warning(result.response)

                    st.markdown("### Classification")
                    c1, c2 = st.columns(2)
                    c1.metric("Status", result.status.upper())
                    c2.metric("Request type", result.request_type)
                    st.write("**Product area:**", result.product_area)

                    st.markdown("### Justification")
                    st.write(result.justification)

                    with st.expander("Raw result"):
                        st.json(result.model_dump())
                else:
                    st.warning("Enter a support query first.")
            except Exception as exc:
                st.error(f"Unable to process request: {type(exc).__name__}")

with samples:
    st.subheader("Sample data & agent output")
    st.caption("Loaded from `public/samples/` next to this app.")

    st.markdown("#### Labeled sample set")
    st.write(
        "A small hand-checkable set of tickets with an expected "
        "(human-written) response, product area, status, and request type "
        "alongside the agent's own output on the same tickets."
    )
    sample_in, sample_out = st.tabs(["Expected (input + output)", "Agent output"])
    with sample_in:
        render_sample_csv(
            "sample_support_tickets.csv",
            "Ticket text plus the expected/reference response for each row.",
        )
    with sample_out:
        render_sample_csv(
            "sample_support_tickets_output_5.csv",
            "The agent's status / product_area / response / justification / "
            "request_type for the same tickets, for side-by-side comparison "
            "against the expected column above.",
        )

    st.divider()

    st.markdown("#### Testing set")
    st.write(
        "A larger, unlabeled batch of tickets (no expected response) used "
        "to exercise the agent more broadly, alongside its output on them."
    )
    test_in, test_out = st.tabs(["Input tickets", "Agent output"])
    with test_in:
        render_sample_csv(
            "support_tickets.csv",
            "Raw ticket subject/issue text used as test input.",
        )
    with test_out:
        render_sample_csv(
            "support_tickets_output_5.csv",
            "The agent's output for every ticket in the input set above.",
        )