from dotenv import load_dotenv
import os
import json
import operator
from datetime import datetime
from typing import Literal, Annotated, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Neo4jVector
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# ── few-shot examples ─────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "question": "Show all customers from Brazil",
        "query": "MATCH (c:Customer) WHERE c.nationality = 'BR' RETURN c.name, c.email",
    },
    {
        "question": "Show all customers from Brazil and United Kingdom",
        "query": "MATCH (c:Customer) WHERE c.nationality = 'BR' OR c.nationality = 'UK' RETURN c.name, c.email",
    },
    {
        "question": "List all business customers with pending KYC status",
        "query": "MATCH (c:Customer) WHERE c.type = 'corporate' AND c.kyc_status = 'pending' RETURN c.name, c.email, c.kyc_status",
    },
    {
        "question": "List all business customers with non rejected KYC status",
        "query": "MATCH (c:Customer) WHERE c.type = 'corporate' AND c.kyc_status <> 'rejected' RETURN c.name, c.email, c.kyc_status",
    },
    {
        "question": "Find all accounts belonging to nicolas silva",
        "query": "MATCH (c:Customer {name:'Nicolas Silva'})-[:HAS_ACCOUNT]->(a:Account) RETURN a",
    },
    {
        "question": "Show me complete info on Nicolas Silva",
        "query": "MATCH (c:Customer {name:'Nicolas Silva'}) RETURN c.id, c.name, c.email, c.nationality, c.kyc_status, c.risk_level, c.type, c.date_of_birth, c.phone, c.tax_id",
    },
    {
        "question": "Show transactions made by Nicolas Silva",
        "query": "MATCH (c:Customer {name:'Nicolas Silva'})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction) RETURN t",
    },
    {
        "question": "Show transactions made by Nicolas Silva in 2023",
        "query": "MATCH (c:Customer {name:'Nicolas Silva'})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction) WHERE t.date >= datetime('2023-01-01') AND t.date <= datetime('2023-12-31') RETURN t",
    },
    {
        "question": "Show transactions made by Brazilian customers",
        "query": "MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction) WHERE c.nationality = 'BR' RETURN c.name, t",
    },
    {
        "question": "Show transactions made by Brazilian business customers",
        "query": "MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction) WHERE c.nationality = 'BR' AND c.type = 'corporate' RETURN c.name, t",
    },
    {
        "question": "Show transactions made by Brazilian business customers with pending KYC",
        "query": "MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction) WHERE c.nationality = 'BR' AND c.type = 'corporate' AND c.kyc_status = 'pending' RETURN c.name, c.email, t",
    },
    {
        "question": "How many british customers do we have?",
        "query": "MATCH (c:Customer) WHERE c.nationality = 'UK' RETURN COUNT(DISTINCT c) AS customer_count",
    },
    {
        "question": "Find customers who share the same address as Nicolas Silva",
        "query": "MATCH (c:Customer {name:'Nicolas Silva'})-[:LIVES_AT]->(a:Address)<-[:LIVES_AT]-(other:Customer) RETURN other.name, other.email",
    },
    {
        "question": "Find customers that share bank accounts with Nicolas Silva",
        "query": "MATCH (c:Customer {name:'Nicolas Silva'})-[:HAS_ACCOUNT]->(a:Account)<-[:HAS_ACCOUNT]-(other:Customer) WHERE other.name <> c.name RETURN other.name, other.email",
    },
    {
        "question": "Find customers that transacted with the same counterparties as Nicolas Silva",
        "query": "MATCH (c:Customer {name:'Nicolas Silva'})-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction)-[:RECEIVED]->(r:Account)<-[:HAS_ACCOUNT]-(other:Customer) WHERE other.name <> c.name RETURN DISTINCT other.name, other.email",
    },
    {
        "question": "Show customers that appear on sanctions lists",
        "query": "MATCH (c:Customer)-[:MATCHES_SANCTION]->(s:Sanction) RETURN c.name, s",
    },
    {
        "question": "Show business customers that appear on sanctions lists",
        "query": "MATCH (c:Customer)-[:MATCHES_SANCTION]->(s:Sanction) WHERE c.type = 'corporate' RETURN c.name, c.email, s",
    },
    {
        "question": "Show customers that are politically exposed persons",
        "query": "MATCH (c:Customer)-[:IN_PEP]->(p:PEP) RETURN c.name, p",
    },
    {
        "question": "Show high risk customers",
        "query": "MATCH (c:Customer) WHERE c.risk_level = 'high' RETURN c.name, c.email, c.risk_level",
    },
    {
        "question": "Show high risk customers with sanctions",
        "query": "MATCH (c:Customer)-[:MATCHES_SANCTION]->(s:Sanction) WHERE c.risk_level = 'high' RETURN c.name, c.email, s",
    },
    {
        "question": "Show high risk customers with transactions in the last week",
        "query": "MATCH (c:Customer)-[:HAS_ACCOUNT]->(a:Account)-[:SENT]->(t:Transaction) WHERE c.risk_level = 'high' AND t.date >= datetime() - duration('P7D') RETURN c.name, c.email, t",
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────
def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def safe_json(obj):
    return json.loads(json.dumps(obj, default=json_serializer))


# ── state ─────────────────────────────────────────────────────────────────────
class KYCState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    pending_query: str
    permission_granted: bool
    query_results: Annotated[list, operator.add]
    web_search_results: str
    web_search_query: str
    next_action: str  # "query" | "clarify" | "search" | "answer"
    iteration_count: int


# ── structured output models ──────────────────────────────────────────────────
class PlanDecision(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about what to do next")
    next_action: Literal["query", "clarify", "answer"] = Field(
        description="'query' to run a graph query, 'clarify' if the question is ambiguous, 'answer' if enough info available"
    )
    plan: str = Field(description="Describe what to query or what to clarify or the answer to give")


class AnalyzeDecision(BaseModel):
    reasoning: str = Field(description="Analysis of the query results")
    needs_more_data: bool = Field(description="True if more graph queries are needed")
    next_query_goal: str = Field(default="", description="What to query next (if needs_more_data)")
    final_answer: str = Field(default="", description="Full answer for the user (if not needs_more_data)")


class ClarificationDecision(BaseModel):
    question: str = Field(description="The clarifying question to ask the user")
    needs_web_search: bool = Field(default=False, description="True if web search would help clarify before asking")
    search_query: str = Field(default="", description="Search query if needs_web_search")


# ── node factories ────────────────────────────────────────────────────────────
def _make_plan_node(model, graph_schema: str):
    planner = model.with_structured_output(PlanDecision)

    def plan_node(state: KYCState) -> dict:
        iteration = state.get("iteration_count", 0)
        print(f"\n[PLAN] Iteration {iteration}/6 — deciding next action")
        query_results_summary = json.dumps(
            state.get("query_results") or [], default=json_serializer
        )[:4000]  # keep context manageable
        web_results = state.get("web_search_results") or ""

        system_prompt = f"""You are a KYC/AML analyst assistant. Decide the next step to answer the user's question.

Graph schema:
{graph_schema}

Query results gathered so far:
{query_results_summary}

Web search context:
{web_results}

Current iteration: {iteration}/6 (stop querying at 6)

Rules:
- Choose "query" if you need to run a Cypher query to get data.
- Choose "clarify" only if the question is genuinely ambiguous (missing customer name, date range, etc.).
- Choose "answer" if you have enough data to write a comprehensive response.
- At iteration 6+, always choose "answer"."""

        decision = planner.invoke(
            [SystemMessage(content=system_prompt)] + list(state.get("messages") or [])
        )
        print(f"[PLAN] Decision: next_action={decision.next_action!r}, plan={decision.plan[:100]}")
        return {"plan": decision.plan, "next_action": decision.next_action}

    return plan_node


def _make_query_maker_node(model, embeddings, neo4j_config: dict, graph_schema: str):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        EXAMPLES,
        embeddings,
        Neo4jVector,
        url=neo4j_config["uri"],
        username=neo4j_config["username"],
        password=neo4j_config["password"],
        index_name="kyc_examples_litellm",
        k=3,
        input_keys=["question"],
    )

    def query_maker_node(state: KYCState) -> dict:
        question = state.get("plan", "")
        print(f"\n[QUERY_MAKER] Building Cypher for goal: {question[:120]}")

        # Build prompt manually — do NOT use FewShotPromptTemplate.format() because
        # Python str.format() trips on Cypher property syntax like {name:'Nicolas Silva'}.
        selected = example_selector.select_examples({"question": question})
        examples_text = "\n\n".join(
            f"User input: {ex['question']}\nCypher query: {ex['query']}"
            for ex in selected
        )
        prompt_text = (
            "You are a Neo4j expert. Given an input question, create a syntactically "
            "correct Cypher query to run. Return ONLY the raw Cypher — no markdown, "
            "no explanation.\n\n"
            f"Schema:\n{graph_schema}\n\n"
            f"Examples:\n{examples_text}\n\n"
            f"User input: {question}\nCypher query:"
        )
        print(f"[QUERY_MAKER] Selected {len(selected)} examples from vector store")

        response = model.invoke([HumanMessage(content=prompt_text)])
        cypher = response.content.strip()
        # Strip markdown code fences if present
        if cypher.startswith("```"):
            lines = cypher.splitlines()
            cypher = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher
        print(f"[QUERY_MAKER] Generated Cypher:\n{cypher}\n")
        return {"pending_query": cypher}

    return query_maker_node


def _user_permission_node(state: KYCState) -> dict:
    query = state.get("pending_query", "")
    print(f"\n[USER_PERMISSION] Pausing for user approval of query:\n{query}\n")
    result = interrupt(
        {
            "type": "permission",
            "query": query,
            "plan": state.get("plan", ""),
        }
    )
    approved = result.get("approved", False) if isinstance(result, dict) else bool(result)
    print(f"[USER_PERMISSION] User {'APPROVED' if approved else 'DENIED'} the query")
    return {"permission_granted": approved}


def _make_query_executer_node(neo4j_graph):
    def query_executer_node(state: KYCState) -> dict:
        cypher = state.get("pending_query", "")
        print(f"\n[QUERY_EXECUTER] Running:\n{cypher}")
        try:
            raw = neo4j_graph.query(cypher)
            results = safe_json(raw)
            print(f"[QUERY_EXECUTER] Got {len(results)} result(s)")
        except Exception as e:
            results = [{"error": str(e)}]
            print(f"[QUERY_EXECUTER] ERROR: {e}")

        return {
            "query_results": [{"query": cypher, "results": results}],
            "iteration_count": (state.get("iteration_count") or 0) + 1,
        }

    return query_executer_node


def _make_analyze_node(model, graph_schema: str):
    analyzer = model.with_structured_output(AnalyzeDecision)

    def analyze_node(state: KYCState) -> dict:
        iteration = state.get("iteration_count", 0)
        query_results = json.dumps(
            state.get("query_results") or [], default=json_serializer
        )[:6000]

        messages = list(state.get("messages") or [])
        original_question = messages[-1].content if messages else ""

        system_prompt = f"""You are a KYC/AML analyst. Analyze the query results and decide:
- If the data is sufficient, write a comprehensive final answer.
- If more data is needed (and iteration < 6), describe what else to query.

Graph schema: {graph_schema}
Iteration: {iteration}/6

Structure your final answer with:
1. Summary (table/list of key findings)
2. Explanation (what it means for KYC/AML compliance)
3. Key Observations (up to 3)
4. Suggested follow-ups"""

        context = f"Original question: {original_question}\n\nQuery results:\n{query_results}"

        print(f"\n[ANALYZE] Iteration {iteration}/6 — reviewing query results")
        decision = analyzer.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=context)]
        )
        print(f"[ANALYZE] needs_more_data={decision.needs_more_data}, reasoning={decision.reasoning[:100]}")

        if not decision.needs_more_data or iteration >= 6:
            answer = decision.final_answer or "No results found."
            print(f"[ANALYZE] Producing final answer ({len(answer)} chars)")
            return {
                "next_action": "answer",
                "messages": [AIMessage(content=answer)],
            }
        print(f"[ANALYZE] Requesting another query: {decision.next_query_goal[:100]}")
        return {"next_action": "query", "plan": decision.next_query_goal}

    return analyze_node


def _make_clarification_node(model):
    clarifier = model.with_structured_output(ClarificationDecision)

    def ask_user_clarification_node(state: KYCState) -> dict:
        system_prompt = (
            "You are a KYC assistant. The user's request is unclear. "
            "Generate a specific clarifying question. "
            "If public information (e.g., what a sanctions list is) would help first, set needs_web_search=True."
        )
        decision = clarifier.invoke(
            [SystemMessage(content=system_prompt)] + list(state.get("messages") or [])
        )

        if decision.needs_web_search:
            return {
                "next_action": "search",
                "web_search_query": decision.search_query,
            }

        # Pause and ask user
        user_answer = interrupt(
            {"type": "clarification", "question": decision.question}
        )
        return {
            "messages": [
                AIMessage(content=decision.question),
                HumanMessage(content=str(user_answer)),
            ],
            "next_action": "query",
        }

    return ask_user_clarification_node


def _make_web_search_node():
    try:
        from duckduckgo_search import DDGS
        _DDGS = DDGS
    except ImportError:
        _DDGS = None

    def web_search_node(state: KYCState) -> dict:
        if _DDGS is None:
            return {
                "web_search_results": "Web search unavailable (install duckduckgo-search)",
                "next_action": "query",
            }
        query = state.get("web_search_query") or ""
        try:
            with _DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=3))
            formatted = "\n".join(f"- {h['title']}: {h['body']}" for h in hits)
        except Exception as e:
            formatted = f"Search failed: {e}"

        return {"web_search_results": formatted, "next_action": "query"}

    return web_search_node


# ── routing ───────────────────────────────────────────────────────────────────
def _route_from_plan(state: KYCState):
    action = state.get("next_action", "answer")
    if action == "query":
        return "query_maker"
    if action == "clarify":
        return "ask_user_clarification"
    return END


def _route_from_permission(state: KYCState):
    if state.get("permission_granted"):
        return "query_executer"
    return "plan"


def _route_from_analyze(state: KYCState):
    if state.get("next_action") == "query":
        return "plan"
    return END


def _route_from_clarification(state: KYCState):
    if state.get("next_action") == "search":
        return "web_search"
    return END


# ── public API ────────────────────────────────────────────────────────────────
def create_agent(model, embeddings, neo4j_graph):
    """Build and compile the KYC LangGraph agent.

    Args:
        model: A LangChain chat model (e.g. ChatOpenAI pointing at LiteLLM proxy).
        embeddings: A LangChain embeddings model (e.g. OpenAIEmbeddings via LiteLLM proxy).
        neo4j_graph: An initialised langchain_neo4j.Neo4jGraph instance.

    Returns:
        Compiled LangGraph with MemorySaver checkpointer.
    """
    graph_schema = neo4j_graph.schema
    neo4j_config = {
        "uri": os.environ["NEO4J_URI"],
        "username": os.environ["NEO4J_USERNAME"],
        "password": os.environ["NEO4J_PASSWORD"],
    }

    plan_node = _make_plan_node(model, graph_schema)
    query_maker_node = _make_query_maker_node(model, embeddings, neo4j_config, graph_schema)
    query_executer_node = _make_query_executer_node(neo4j_graph)
    analyze_node = _make_analyze_node(model, graph_schema)
    clarification_node = _make_clarification_node(model)
    web_search_node = _make_web_search_node()

    builder = StateGraph(KYCState)

    builder.add_node("plan", plan_node)
    builder.add_node("query_maker", query_maker_node)
    builder.add_node("user_permission", _user_permission_node)
    builder.add_node("query_executer", query_executer_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("ask_user_clarification", clarification_node)
    builder.add_node("web_search", web_search_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _route_from_plan, ["query_maker", "ask_user_clarification", END])
    builder.add_edge("query_maker", "user_permission")
    builder.add_conditional_edges("user_permission", _route_from_permission, ["query_executer", "plan"])
    builder.add_edge("query_executer", "analyze")
    builder.add_conditional_edges("analyze", _route_from_analyze, ["plan", END])
    builder.add_conditional_edges("ask_user_clarification", _route_from_clarification, ["web_search", END])
    builder.add_edge("web_search", "plan")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
