import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
from dotenv import load_dotenv
from memory import store
from agent.research_agent import DeepResearchAgent
import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging

logging.set_verbosity_error()

torch.set_num_threads(2)
load_dotenv()
conn = store.conn(Path("session.db"))

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔎",
    layout="centered"
)

st.markdown("# 🔎 Deep Research Agent")
st.caption("Grounded answers with citations from live web sources.")
st.divider()

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

if "session_id" not in st.session_state:
    st.session_state.session_id = store.new_session(conn)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = DeepResearchAgent(conn,load_sentence_model())

with st.sidebar:
    st.markdown("### Session")
    st.caption(f"`{st.session_state.session_id[:8]}…`")

    if st.button("➕ New Session"):
        st.session_state.session_id = store.new_session(conn)
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. 🗺️ Plans search queries\n"
        "2. 🔎 Searches via Tavily\n"
        "3. 📄 Fetches source pages\n"
        "4. ✂️ Selects best snippets\n"
        "5. ✍️ Generates cited answer"
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


STEP_LABELS = {
    "planning":  "🗺️  Planning research strategy",
    "searching": "🔎  Searching the web",
    "fetching":  "📄  Fetching sources",
    "selecting": "✂️  Selecting relevant context",
    "answering": "✍️  Generating answer with citations",
}


query = st.chat_input("Ask a research question...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)


    with st.chat_message("assistant"):
        status = st.empty()
        answer_box = st.empty()

        agent = st.session_state.agent
        final_answer = ""

        for event in agent.run(st.session_state.session_id, query):
            step = event["step"]

            if step != "done":
                label = STEP_LABELS.get(step, step)
                status.info(f"{label}  \n`{event['data']}`")

            else:
                status.empty()
                final_answer = event["answer"]
                answer_box.markdown(final_answer)

        if event.get("snippets"):
            with st.expander("📚 Sources used"):
                for i, s in enumerate(event["snippets"], 1):
                    st.markdown(f"**{i}. [{s.title}]({s.url})**  \n`{s.domain}`")

    with st.expander("🧠 Research Trace"):

        st.markdown("### 🔎 Search Queries")
        for q in event.get("search_queries", []):
            st.code(q)

        st.markdown("### 🌐 URLs Opened")
        for url in event.get("urls", []):
            st.markdown(f"- {url}")

        st.markdown("### ✂️ Selected Context Snippets")

        for i, s in enumerate(event.get("snippets", []), 1):
            with st.expander(f"Snippet {i} — {s.title}"):

                st.markdown(f"### Source")
                st.markdown(f"[{s.title}]({s.url})")

                st.markdown("### URL")
                st.code(s.url)

                st.markdown("### Full Snippet")
                st.write(s.text)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})