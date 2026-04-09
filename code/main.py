# streamlit interface for asking questions about the graph
from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agentic_ver import create_agent
from semantic_ver import create_chain
from tools import graph, set_tools
import uuid
import json
import os
from pathlib import Path
import time
from datetime import datetime

# JSON serializer for non-standard types
def json_serializer(obj):
    """Serialize non-standard JSON types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)
    
# Streamlit UI
st.set_page_config(page_title=" Assistant", layout="wide")
st.title("💬 Ask me anything related to KYC")

# Conversation persistence functions
CONVERSATIONS_DIR = Path("conv")

def ensure_conversations_dir():
    """Ensure the conversations directory exists."""
    CONVERSATIONS_DIR.mkdir(exist_ok=True)

def load_conversations_from_disk():
    """Load all conversations from disk."""
    ensure_conversations_dir()
    conversations = {}
    conv_counter = 0
    
    for file in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                conv_id = data.get("id")
                if conv_id:
                    conversations[conv_id] = {
                        "name": data.get("name", f"Conversation {conv_counter + 1}"),
                        "messages": data.get("messages", [])
                    }
                    conv_counter = max(conv_counter, int(data.get("counter", 0)))
        except Exception as e:
            st.warning(f"Error loading conversation from {file}: {e}")
    
    return conversations, conv_counter

def save_conversations_to_disk(conversations, conv_counter):
    """Save all conversations to disk."""
    ensure_conversations_dir()
    for conv_id, data in conversations.items():
        file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
        try:
            with open(file_path, "w") as f:
                json.dump({
                    "id": conv_id,
                    "name": data["name"],
                    "messages": data["messages"],
                    "counter": conv_counter
                }, f, indent=2, default=json_serializer)
        except Exception as e:
            st.error(f"Error saving conversation {conv_id}: {e}")


@st.cache_resource
def load_agent(mode):
    # build model
    model= ChatOllama(model="deepseek-v3.1:671b-cloud", temperature=0)
    query_model = ChatOllama(model="ministral-3:14b-cloud", temperature=0)
    if mode == "agentic":
        tools = set_tools
        agent = create_agent(model, tools)
    else:
        agent = create_chain(model,query_model, graph)
    return agent

agent = load_agent("chain")

# Initialize conversations from disk if not present
if "conversations" not in st.session_state:
    st.session_state.conversations, st.session_state.conv_counter = load_conversations_from_disk()

# If no conversations exist, create first one
if not st.session_state.conversations:
    st.session_state.conv_counter = 1
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"name": f"Conversation {st.session_state.conv_counter}", "messages": []}
    st.session_state.current_conv = conv_id
    save_conversations_to_disk(st.session_state.conversations, st.session_state.conv_counter)

if "current_conv" not in st.session_state or st.session_state.current_conv not in st.session_state.conversations:
    # Set current conversation to first one
    st.session_state.current_conv = list(st.session_state.conversations.keys())[0]

# Sidebar for conversation management
with st.sidebar:
    st.header("Conversations")
    
    # Button to start new conversation
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.conv_counter += 1
        conv_id = str(uuid.uuid4())
        st.session_state.conversations[conv_id] = {"name": f"Conversation {st.session_state.conv_counter}", "messages": []}
        st.session_state.current_conv = conv_id
        save_conversations_to_disk(st.session_state.conversations, st.session_state.conv_counter)
        st.rerun()
    
    st.divider()
    
    # Display list of conversations
    for conv_id, data in st.session_state.conversations.items():
        # Highlight current conversation
        is_current = conv_id == st.session_state.current_conv
        button_label = f"✓ {data['name']}" if is_current else data['name']
        button_style = "primary" if is_current else "secondary"
        
        if st.button(button_label, use_container_width=True, key=f"conv_{conv_id}"):
            st.session_state.current_conv = conv_id
            st.rerun()

# Get current conversation messages
current_messages = st.session_state.conversations[st.session_state.current_conv]["messages"]

# Display chat history for current conversation
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show Query Details for assistant messages if available
        if message["role"] == "assistant" and "intermediate_steps" in message:
            with st.expander("Query Details", expanded=False):
                intermediate = message["intermediate_steps"]
                if isinstance(intermediate, list) and intermediate:
                    for step in intermediate:
                        if isinstance(step, dict):
                            if 'query' in step:
                                st.subheader('Generated Cypher Query')
                                st.code(step['query'], language='cypher')
                            if 'context' in step:
                                st.subheader('Cypher Result Context')
                                st.json(step['context'])
                    if not any(isinstance(step, dict) and ('query' in step or 'context' in step) for step in intermediate):
                        st.json(intermediate)
                else:
                    st.json(intermediate)

# User input
prompt = st.chat_input("Ask about the graph...")

if prompt:
    # Show user message
    current_messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent (use last 5 messages from current conversation for context)
    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            history_messages = []
            for msg in current_messages[-5:]:  # Last 5 messages
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                else:
                    history_messages.append(AIMessage(content=msg["content"]))

            query = "\n".join([msg.content for msg in history_messages[-5:]])  # Last 5 messages content
            
            # Timing instrumentation
            start_time = time.time()
            result = agent.invoke({"query": query})
            total_time = time.time() - start_time
            
            answer = result.get('result') or result.get('output_text') or ""
            
            st.markdown(answer)
            
            # Show timing info
            st.caption(f"⏱️ Response generated in {total_time:.2f}s")

            intermediate = result.get('intermediate_steps')
            with st.expander("Query Details", expanded=False):
                if isinstance(intermediate, list) and intermediate:
                    for step in intermediate:
                        if isinstance(step, dict):
                            if 'query' in step:
                                st.subheader('Generated Cypher Query')
                                st.code(step['query'], language='cypher')
                            if 'context' in step:
                                st.subheader('Cypher Result Context')
                                st.json(step['context'])
                    if not any(isinstance(step, dict) and ('query' in step or 'context' in step) for step in intermediate):
                        st.json(intermediate)
                else:
                    st.json(intermediate)

    # Save assistant reply to current conversation with intermediate steps
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "intermediate_steps": intermediate
    }
    current_messages.append(assistant_message)
    
    # Persist conversations to disk
    save_conversations_to_disk(st.session_state.conversations, st.session_state.conv_counter)
