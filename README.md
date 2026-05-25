# Agentic AI Workflows

This repository is a cleaned-up workspace for agentic systems built with
LangGraph, LangChain, Ollama, and a memory-persistent Streamlit experience.
It combines experimental notebooks with runnable Python projects so the codebase
is easier to navigate, extend, and present professionally.

<p align="center">
  <img src="assets/streamlit_ui_longtermMemory.png" alt="Memory persistent Streamlit chatbot" width="900">
</p>

## Highlights

- Memory-persistent Streamlit chatbot backed by PostgreSQL + LangGraph.
- LangGraph workflow experiments for sequential, conditional, parallel, and iterative orchestration.
- RAG and tool-use examples for documents, PDFs, and structured outputs.
- Notebook exploration kept separate from runnable project code.

## Repository Layout

- `projects/` - runnable scripts and app entry points.
- `notebooks/` - exploratory notebooks and proof-of-concept work.
- `assets/` - screenshots and diagrams used in documentation.
- `data/` - local PDFs and other data sources referenced by the workflows.
- `tools/` - maintenance utilities such as repository reorganizers.

## Featured Project

### Long-Term Memory Chatbot UI

The Streamlit app in `longtermMemory_Chatbot_UI.py` is the showcase project in
this repo. It uses a LangGraph checkpointer with PostgreSQL persistence so the
conversation survives refreshes and can be separated by thread ID.

<p align="center">
  <img src="assets/streamlit_ui_longtermMemory_db.png" alt="Memory database view" width="900">
</p>

## Training / Workflow Overview

The repository also includes a visual training/interaction flow for the LLM
stack.

<p align="center">
  <img src="llm_training_loop.png" alt="LLM training loop" width="900">
</p>

## Quick Start

1. Install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows PowerShell
pip install --upgrade pip
pip install -e .
```

2. Configure environment variables.

Create a `.env` file with your database and model settings, for example:

```env
db_url=postgresql://user:password@localhost:5432/agentic_memory
```

3. Run the memory chatbot.

```bash
python main.py
# or
streamlit run longtermMemory_Chatbot_UI.py
```

4. Open the notebooks in `notebooks/` for the LangGraph experiments.

## Notes

- Large model checkpoints and generated outputs should stay out of Git.
- If you add a new workflow, place the reusable code in `projects/` and keep
  the notebook version in `notebooks/`.
- The repository is intentionally organized for presentation, reproducibility,
  and easier maintenance.
# Agentic AI Workspace 🤖

A specialized repository for building, testing, and deploying **Stateful Multi-Agent Systems** using LangGraph, LangChain, and Local LLMs.

## 🚀 Overview
This repository serves as a professional laboratory for **Agentic Design Patterns**. The focus is on moving beyond linear chains into cyclic, self-correcting, and autonomous workflows that leverage local compute.

## 🛠️ Tech Stack
* **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful orchestration)
* **Framework:** [LangChain](https://www.langchain.com/)
* **Local Inference:** [Ollama](https://ollama.com/) (Running Qwen2.5-Coder & Phi-3.5)
* **Environment:** Arch Linux / Windows (Conda)
* **Memory Layers:** PostgreSQL (Persistence) & Obsidian (Semantic Knowledge)