import os, re, sys, asyncio, sqlite3, json
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import create_model
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = os.environ.get("MODEL_NAME", "llama3.2:3b")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DB = "movies.db"

_SKILL_FILE = os.path.join(os.path.dirname(__file__), "SKILL.md")
_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(os.path.dirname(__file__), "tools.py")],
)

_TYPE_MAP = {"string": str, "integer": int, "number": float, "boolean": bool}


def _load_instructions(path: str) -> str:
    with open(path) as f:
        content = f.read()
    match = re.search(r"## Instructions\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    return match.group(1).strip() if match else ""


def _load_schema() -> str:
    conn = sqlite3.connect(DB)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    lines = []
    for t in tables:
        cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
        col_str = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
        lines.append(f"  {t}: {col_str}")
    conn.close()
    return "\n".join(lines)


def _clean_schema(s):
    keep = {"type", "properties", "required", "description", "items", "enum", "anyOf"}
    out = {k: v for k, v in s.items() if k in keep}
    if "properties" in out:
        out["properties"] = {k: _clean_schema(v) for k, v in out["properties"].items()}
    return out


def _schema_to_model(name: str, schema: dict):
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = {}
    for field_name, prop in props.items():
        py_type = _TYPE_MAP.get(prop.get("type", "string"), str)
        fields[field_name] = (py_type, ...) if field_name in required else (Optional[py_type], None)
    return create_model(f"{name}Args", **fields)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _call_tool(name: str, args: dict) -> str:
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            return result.content[0].text if result.content else ""


async def _list_mcp_tools():
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            return (await session.list_tools()).tools


def _load_mcp_tools():
    mcp_tools = _run(_list_mcp_tools())
    tools = []
    for t in mcp_tools:
        clean = _clean_schema(t.inputSchema)
        model = _schema_to_model(t.name, clean)

        def make_fn(tool_name):
            def fn(**kwargs):
                return _run(_call_tool(tool_name, kwargs))
            fn.__name__ = tool_name
            return fn

        tools.append(StructuredTool(
            name=t.name,
            description=t.description or t.name,
            func=make_fn(t.name),
            args_schema=model,
        ))
    return tools


SYSTEM = (
    f"You are a movie database assistant.\n\n"
    f"{_load_instructions(_SKILL_FILE)}\n\n"
    f"Database schema (exact column names — use these, do not guess):\n"
    f"{_load_schema()}"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    tools = _load_mcp_tools()
    llm = ChatOllama(model=MODEL, base_url=OLLAMA_URL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: State):
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM)] + list(msgs)
        return {"messages": [llm_with_tools.invoke(msgs)]}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()
