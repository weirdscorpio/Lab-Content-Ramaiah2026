import os, sys, json, asyncio
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL_NAME", "llama3.2:3b")

_SERVER = StdioServerParameters(
    command=sys.executable,
    args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")],
)

SYSTEM = (
    "You are a helpful assistant for a bookstore SQLite database. "
    "Use the available tools to look up real data before answering. "
    "Tables: authors, books, sales. "
    "Always query instead of guessing. Format results clearly."
)


# ── async helpers ─────────────────────────────────────────────────────────

def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _list_tools():
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return (await s.list_tools()).tools


async def _call_tool(name, args):
    async with stdio_client(_SERVER) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(name, args)
            return res.content[0].text if res.content else "{}"


def _clean_schema(s):
    if not isinstance(s, dict):
        return s
    keep = {"type", "properties", "required", "description", "items", "enum", "anyOf"}
    out = {k: v for k, v in s.items() if k in keep}
    if "properties" in out:
        out["properties"] = {k: _clean_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _clean_schema(out["items"])
    return out


def _tools_for_ollama(tools):
    return [
        {"type": "function", "function": {
            "name": t.name,
            "description": t.description,
            "parameters": _clean_schema(t.inputSchema),
        }}
        for t in tools
    ]


# ── routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/schema")
def schema():
    try:
        tables_raw = _run(_call_tool("list_tables", {}))
        tables = json.loads(tables_raw)
        result = {}
        for t in tables:
            cols_raw = _run(_call_tool("describe_table", {"table": t}))
            result[t] = json.loads(cols_raw)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json()
    user_msg = body.get("message", "").strip()
    history = body.get("history", [])
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    tools = _run(_list_tools())
    ollama_tools = _tools_for_ollama(tools)
    messages = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": user_msg}]

    def generate():
        try:
            r1 = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": MODEL, "messages": messages, "tools": ollama_tools, "stream": False},
                timeout=60,
            )
            if not r1.ok:
                raise RuntimeError(f"Ollama {r1.status_code}: {r1.text}")

            assistant_msg = r1.json().get("message", {})
            tool_calls = assistant_msg.get("tool_calls") or []

            if tool_calls:
                conv = messages + [assistant_msg]
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': name, 'args': args})}\n\n"
                    result = _run(_call_tool(name, args))
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': name, 'result': result})}\n\n"
                    conv.append({"role": "tool", "content": result})

                r2 = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": MODEL, "messages": conv, "stream": True},
                    stream=True, timeout=60,
                )
                r2.raise_for_status()
                for line in r2.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    if chunk.get("done"):
                        yield "data: [DONE]\n\n"
                        return
            else:
                content = assistant_msg.get("content", "")
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/health")
def health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return jsonify({"ollama": "ok", "models": models})
    except Exception as e:
        return jsonify({"ollama": "error", "detail": str(e)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
