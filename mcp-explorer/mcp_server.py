import sqlite3, json
from mcp.server.fastmcp import FastMCP

DB = "bookstore.db"
mcp = FastMCP("bookstore")


def qry(sql, params=()):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def exe(sql, params=()):
    conn = sqlite3.connect(DB)
    cur = conn.execute(sql, params)
    conn.commit()
    res = {"rows_affected": cur.rowcount, "last_insert_id": cur.lastrowid}
    conn.close()
    return res


@mcp.tool()
def list_tables() -> str:
    """List all tables in the database."""
    rows = qry("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return json.dumps([r["name"] for r in rows])


@mcp.tool()
def describe_table(table: str) -> str:
    """Get column names and types for a table."""
    return json.dumps(qry(f"PRAGMA table_info({table})"))


@mcp.tool()
def run_query(sql: str) -> str:
    """Execute a SQL SELECT query and return rows as JSON."""
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries allowed"})
    return json.dumps(qry(sql), default=str)


@mcp.tool()
def insert_row(table: str, data: dict) -> str:
    """Insert a row into a table. data is a dict of column->value pairs."""
    cols = ", ".join(data.keys())
    ph = ", ".join("?" * len(data))
    return json.dumps(exe(f"INSERT INTO {table} ({cols}) VALUES ({ph})", list(data.values())))


if __name__ == "__main__":
    mcp.run()
