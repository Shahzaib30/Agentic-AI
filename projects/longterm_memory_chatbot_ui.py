"""Memory-persistent Streamlit chatbot powered by LangGraph and PostgreSQL."""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import MessagesState, START, StateGraph
from psycopg_pool import ConnectionPool


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
DEFAULT_THREAD_ID = "default_thread"


def get_database_url() -> str | None:
    load_dotenv()
    return os.getenv("db_url") or os.getenv("DATABASE_URL")


@st.cache_resource
def get_pool(db_url: str) -> ConnectionPool:
    return ConnectionPool(conninfo=db_url, max_size=5, kwargs={"autocommit": True})


@st.cache_resource
def build_graph(model_name: str) -> StateGraph:
    llm = ChatOllama(model=model_name, temperature=0.7, streaming=True)

    def chat_node(state: MessagesState, config: RunnableConfig) -> MessagesState:
        response = llm.invoke(state["messages"], config=config)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    return builder


def ensure_session_state() -> None:
    if "current_thread" not in st.session_state:
        st.session_state.current_thread = DEFAULT_THREAD_ID


def create_new_chat() -> None:
    st.session_state.current_thread = str(uuid.uuid4())


def render_sidebar() -> None:
    st.sidebar.title("Agent Controls")
    st.sidebar.caption("Switch thread IDs to simulate different long-term memories.")
    st.sidebar.button(
        "➕ New Chat",
        help="Create a new conversation thread.",
        on_click=create_new_chat,
    )
    st.sidebar.text_input("Active Thread ID", key="current_thread")


def render_history(graph, config) -> None:
    current_state = graph.get_state(config)
    messages = current_state.values.get("messages", [])
    for message in messages:
        if message.type == "human":
            st.chat_message("user").write(message.content)
        elif message.type == "ai":
            st.chat_message("assistant").write(message.content)


def stream_answer(graph, config, user_input: str):
    input_state = {"messages": [("user", user_input)]}
    for chunk, _ in graph.stream(input_state, config, stream_mode="messages"):
        if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.content:
            if isinstance(chunk.content, str):
                yield chunk.content


def main() -> None:
    st.set_page_config(page_title="Long Term Memory Chatbot", page_icon=":robot_face:", layout="wide")
    ensure_session_state()

    st.title("LangGraph Long Term Memory Chatbot")
    st.write(
        "This app stores conversation state in PostgreSQL using a LangGraph "
        "checkpointer, so refreshes do not lose the thread history."
    )

    db_url = get_database_url()
    if not db_url:
        st.error("Missing database URL. Set `db_url` or `DATABASE_URL` in your environment or `.env` file.")
        st.stop()

    render_sidebar()

    config = {"configurable": {"thread_id": st.session_state.current_thread}}
    pool = get_pool(db_url)
    graph_builder = build_graph(DEFAULT_MODEL)

    with pool.connection() as conn:
        memory = PostgresSaver(conn)
        memory.setup()
        graph = graph_builder.compile(checkpointer=memory)

        render_history(graph, config)

        if user_input := st.chat_input("Your message"):
            st.chat_message("user").write(user_input)
            with st.chat_message("assistant"):
                st.write_stream(stream_answer(graph, config, user_input))


if __name__ == "__main__":
    main()
