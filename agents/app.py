import os, json, asyncio, queue, threading
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from langchain_core.messages import HumanMessage
from agent import build_graph

app = Flask(__name__)
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

graph = build_graph()


def stream_events(user_message):
    """Run the LangGraph agent in a background thread, yield raw events."""
    q = queue.Queue()

    async def run():
        try:
            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                version="v2",
            ):
                q.put(event)
        except Exception as e:
            q.put({"__error__": str(e)})
        finally:
            q.put(None)

    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()

    while True:
        ev = q.get()
        if ev is None:
            break
        yield ev


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json()
    user_msg = body.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "empty"}), 400

    def generate():
        try:
            for event in stream_events(user_msg):
                if "__error__" in event:
                    yield f"data: {json.dumps({'type': 'error', 'message': event['__error__']})}\n\n"
                    break

                kind = event.get("event", "")
                name = event.get("name", "")
                node = event.get("metadata", {}).get("langgraph_node", "")

                if kind == "on_chain_start" and node in ("agent", "tools"):
                    yield f"data: {json.dumps({'type': 'node_start', 'node': node})}\n\n"

                elif kind == "on_chain_end" and node in ("agent", "tools"):
                    yield f"data: {json.dumps({'type': 'node_end', 'node': node})}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                elif kind == "on_tool_start":
                    args = event["data"].get("input", {})
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': name, 'args': args})}\n\n"

                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    result = output.content if hasattr(output, "content") else str(output)
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': name, 'result': result[:600]})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
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


@app.route("/api/schema")
def schema():
    import sqlite3, json as _json
    try:
        conn = sqlite3.connect("movies.db")
        conn.row_factory = sqlite3.Row
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        result = {}
        for t in tables:
            result[t] = [dict(r) for r in conn.execute(f"PRAGMA table_info({t})").fetchall()]
        conn.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5051, debug=False)
