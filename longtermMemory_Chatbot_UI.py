import streamlit as st
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import uuid



# Long Term Memory DB
load_dotenv()
db_url = os.getenv("db_url")
pool = ConnectionPool(conninfo=db_url, max_size=5, kwargs={"autocommit": True})

# UI Config

st.set_page_config(page_title="Long Term Memory Chatbot", page_icon=":robot_face:")

if "current_thread" not in st.session_state:
    st.session_state.current_thread = "default_thread"


# Backend

@st.cache_resource
def backend():
    llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.7)
    
    def chat_node(state: MessagesState) -> MessagesState:
        return {
            "messages" : [llm.invoke(state["messages"])]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")

    return builder

builder = backend()



# UI

def create_new_chat():
    st.session_state.current_thread = str(uuid.uuid4())

st.sidebar.title("Agent Controls")
st.sidebar.write("Change the thread ID to simulate talking to different clients")
    
if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun() 


thread_id = st.sidebar.text_input(
    "Active Thread ID", 
    key="current_thread" 
)

config = {
    "configurable": {
        "thread_id": st.session_state.current_thread
    }
}

st.title("LangGraph Long Term Memory Chatbot")
st.write("If you refresh the page, the chatbot will remember the previous conversation. Change the thread ID to start a new conversation.")

with pool.connection() as conn:
    memory = PostgresSaver(conn)
    memory.setup()

    graph = builder.compile(checkpointer = memory)
    current_state = graph.get_state(config)

    if "messages" in current_state.values:
        for msg in current_state.values["messages"]:
            if msg.type == "human":
                st.chat_message("user").write(msg.content)
            elif msg.type == "ai":
                st.chat_message("assistant").write(msg.content)
    
    if user_input := st.chat_input("Your message"):

        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Bot is typing..."):

                input_state = {
                    "messages": [("user", user_input)]
                }
                result = graph.invoke(input_state, config)
                last_message = result["messages"][-1].content
                st.write(last_message)
