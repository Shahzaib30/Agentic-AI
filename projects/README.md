# Projects

Runnable code for the repository lives here.

Current focus areas:

- `longterm_memory_chatbot_ui.py` - memory-persistent Streamlit chatbot.
- LangGraph workflow examples for sequential, conditional, parallel, and iterative patterns.
- RAG and structured-output experiments that can be promoted into reusable modules.

Legacy scripts have been renamed to clearer snake_case files:

- `basic_chain.py`
- `basic_rag.py`
- `basic_knowledge_chain.py`
- `rag_personal_docs.py`
- `rag_pdf_summarizer.py`
- `structured_output_extraction.py`
- `postgres_tool_agent.py`
- `gpu_stream_test.py`

When adding a new experiment, prefer a script here and keep the notebook in
`notebooks/` only as the exploratory version.
