# streamlit interface for asking questions about the graph
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agentic_ver import create_agent
from semantic_ver import create_chain
from tools import graph, set_tools
    
# Streamlit UI
st.set_page_config(page_title=" Assistant", layout="wide")
st.title("💬 Ask me anything related to KYC")
@st.cache_resource
def load_agent(mode):
    # build model
    model= ChatOllama(model="qwen3.5:cloud", temperature=0)
    if mode == "agentic":
        tools = set_tools
        agent = create_agent(model, tools)
    else:
        agent = create_chain(model, graph)
    return agent

agent = load_agent("chain")
# Store conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# User input
prompt = st.chat_input("Ask about the graph...")

if prompt:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent (use full chat history for context)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                else:
                    history_messages.append(AIMessage(content=msg["content"]))

            #response = agent.invoke({"messages": history_messages})
            query = "\n".join([msg.content for msg in history_messages[-5:]])  # Use the last 5 messages for context
            result = agent.invoke({"query": query})
            answer = result['result']
            #answer = response['result']
            st.markdown(answer)

    # Save assistant reply, and delete the older message if size is greather than 5
    st.session_state.messages.append({"role": "assistant", "content": answer})
