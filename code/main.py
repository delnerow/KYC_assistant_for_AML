import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from agentic_ver import create_agent
from tools import graph as neo4j_graph
import uuid
import json
import os
from pathlib import Path
from datetime import datetime

# ── helpers ───────────────────────────────────────────────────────────────────
def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="KYC Assistant", layout="wide")
st.title("💬 KYC / AML Assistant")

# ── agent init ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    model = ChatOpenAI(
        model=os.getenv("LITELLM_MODEL", "anthropic/claude-sonnet-4-6"),
        base_url=os.getenv("LITELLM_BASE_URL"),
        api_key=os.getenv("LITELLM_API_KEY"),
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=os.getenv("LITELLM_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        base_url=os.getenv("LITELLM_BASE_URL"),
        api_key=os.getenv("LITELLM_API_KEY"),
    )
    return create_agent(model, embeddings, neo4j_graph)


agent = load_agent()

# ── conversation persistence ──────────────────────────────────────────────────
CONVERSATIONS_DIR = Path("conv")


def ensure_conversations_dir():
    CONVERSATIONS_DIR.mkdir(exist_ok=True)


def load_conversations_from_disk():
    ensure_conversations_dir()
    conversations = {}
    conv_counter = 0
    for file in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
            conv_id = data.get("id")
            if conv_id:
                conversations[conv_id] = {
                    "name": data.get("name", f"Conversation {conv_counter + 1}"),
                    "messages": data.get("messages", []),
                    "thread_id": data.get("thread_id", str(uuid.uuid4())),
                }
                conv_counter = max(conv_counter, int(data.get("counter", 0)))
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")
    return conversations, conv_counter


def save_conversations_to_disk(conversations, conv_counter):
    ensure_conversations_dir()
    for conv_id, data in conversations.items():
        file_path = CONVERSATIONS_DIR / f"{conv_id}.json"
        try:
            with open(file_path, "w") as f:
                json.dump(
                    {
                        "id": conv_id,
                        "name": data["name"],
                        "messages": data["messages"],
                        "thread_id": data.get("thread_id", conv_id),
                        "counter": conv_counter,
                    },
                    f,
                    indent=2,
                    default=json_serializer,
                )
        except Exception as e:
            st.error(f"Error saving conversation {conv_id}: {e}")


# ── session state bootstrap ───────────────────────────────────────────────────
if "conversations" not in st.session_state:
    st.session_state.conversations, st.session_state.conv_counter = (
        load_conversations_from_disk()
    )

if not st.session_state.conversations:
    st.session_state.conv_counter = 1
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "name": f"Conversation {st.session_state.conv_counter}",
        "messages": [],
        "thread_id": str(uuid.uuid4()),
    }
    st.session_state.current_conv = conv_id
    save_conversations_to_disk(st.session_state.conversations, st.session_state.conv_counter)

if (
    "current_conv" not in st.session_state
    or st.session_state.current_conv not in st.session_state.conversations
):
    st.session_state.current_conv = list(st.session_state.conversations.keys())[0]

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Conversations")
    if st.button("➕ New Conversation", use_container_width=True):
        st.session_state.conv_counter += 1
        conv_id = str(uuid.uuid4())
        st.session_state.conversations[conv_id] = {
            "name": f"Conversation {st.session_state.conv_counter}",
            "messages": [],
            "thread_id": str(uuid.uuid4()),
        }
        st.session_state.current_conv = conv_id
        save_conversations_to_disk(
            st.session_state.conversations, st.session_state.conv_counter
        )
        st.rerun()

    st.divider()
    for conv_id, data in st.session_state.conversations.items():
        is_current = conv_id == st.session_state.current_conv
        label = f"✓ {data['name']}" if is_current else data["name"]
        if st.button(label, use_container_width=True, key=f"conv_{conv_id}"):
            st.session_state.current_conv = conv_id
            st.rerun()

# ── current conversation refs ─────────────────────────────────────────────────
current_conv_data = st.session_state.conversations[st.session_state.current_conv]
current_messages = current_conv_data["messages"]
thread_id = current_conv_data.get("thread_id", st.session_state.current_conv)
graph_config = {"configurable": {"thread_id": thread_id}}


# ── helper: get pending interrupt ─────────────────────────────────────────────
def get_pending_interrupt():
    """Return the interrupt payload if the graph is suspended, else None."""
    try:
        state = agent.get_state(graph_config)
        if state.tasks:
            interrupts = state.tasks[0].interrupts
            if interrupts:
                return interrupts[0].value
    except Exception:
        pass
    return None


def get_final_answer_from_state():
    """Extract the latest AI message from graph state after completion."""
    try:
        state = agent.get_state(graph_config)
        msgs = state.values.get("messages") or []
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg.content
    except Exception:
        pass
    return None


def get_query_results_from_state():
    """Extract accumulated query results from graph state."""
    try:
        state = agent.get_state(graph_config)
        return state.values.get("query_results") or []
    except Exception:
        return []


# ── permission approval widget ────────────────────────────────────────────────
interrupt_data = get_pending_interrupt()

if interrupt_data and interrupt_data.get("type") == "permission":
    with st.container():
        st.warning("**The agent wants to run a graph query — please review:**")
        st.caption(f"_Plan: {interrupt_data.get('plan', '')}_")
        st.code(interrupt_data.get("query", ""), language="cypher")
        col1, col2 = st.columns(2)
        if col1.button("✅ Approve & Execute", use_container_width=True, key="approve_btn"):
            with st.spinner("Executing query..."):
                agent.invoke(Command(resume={"approved": True}), graph_config)
            # Check whether the graph finished or paused again (another query)
            after_interrupt = get_pending_interrupt()
            if not after_interrupt:
                # Graph completed — grab the answer and persist it
                answer = get_final_answer_from_state() or ""
                query_results = get_query_results_from_state()
                if answer:
                    current_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "query_results": query_results,
                    })
                    save_conversations_to_disk(
                        st.session_state.conversations, st.session_state.conv_counter
                    )
            st.rerun()
        if col2.button("❌ Deny", use_container_width=True, key="deny_btn"):
            with st.spinner("Re-planning..."):
                agent.invoke(Command(resume={"approved": False}), graph_config)
            st.rerun()
    st.divider()

# ── chat history display ──────────────────────────────────────────────────────
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("query_results"):
            with st.expander("Query Details", expanded=False):
                for step in message["query_results"]:
                    if isinstance(step, dict):
                        if "query" in step:
                            st.subheader("Cypher Query")
                            st.code(step["query"], language="cypher")
                        if "results" in step:
                            st.subheader("Results")
                            st.json(step["results"])

# ── chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about KYC/AML data...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    current_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Check if there's a pending clarification interrupt to resume
            pending = get_pending_interrupt()
            if pending and pending.get("type") == "clarification":
                # User's message is the answer to the clarification
                agent.invoke(Command(resume=prompt), graph_config)
            else:
                # New question — start fresh run
                agent.invoke(
                    {
                        "messages": [HumanMessage(content=prompt)],
                        "query_results": [],
                        "iteration_count": 0,
                        "web_search_results": "",
                        "plan": "",
                        "pending_query": "",
                        "permission_granted": False,
                        "next_action": "",
                        "web_search_query": "",
                    },
                    graph_config,
                )

        # Check what happened after invoke
        new_interrupt = get_pending_interrupt()

        if new_interrupt and new_interrupt.get("type") == "permission":
            # Will be handled by the permission widget on next rerun
            st.info("Query ready for review — see approval prompt above.")
            st.rerun()

        elif new_interrupt and new_interrupt.get("type") == "clarification":
            # Show clarification question as assistant message
            question = new_interrupt.get("question", "")
            st.markdown(question)
            current_messages.append(
                {"role": "assistant", "content": question, "query_results": []}
            )

        else:
            # Graph completed — extract final answer
            answer = get_final_answer_from_state() or ""
            query_results = get_query_results_from_state()

            if answer:
                st.markdown(answer)
                if query_results:
                    with st.expander("Query Details", expanded=False):
                        for step in query_results:
                            if isinstance(step, dict):
                                if "query" in step:
                                    st.subheader("Cypher Query")
                                    st.code(step["query"], language="cypher")
                                if "results" in step:
                                    st.subheader("Results")
                                    st.json(step["results"])

                current_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "query_results": query_results,
                    }
                )
            else:
                fallback = "I wasn't able to generate a response. Please try again."
                st.markdown(fallback)
                current_messages.append(
                    {"role": "assistant", "content": fallback, "query_results": []}
                )

    save_conversations_to_disk(
        st.session_state.conversations, st.session_state.conv_counter
    )
