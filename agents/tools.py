import sqlite3, json
from datetime import date
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("movies")
DB = "movies.db"


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
    """List all tables in the movies database."""
    rows = qry("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return json.dumps([r["name"] for r in rows])


@mcp.tool()
def describe_table(table: str) -> str:
    """Get columns and types for a database table."""
    if not table.replace("_", "").isalnum():
        return json.dumps({"error": "Invalid table name"})
    return json.dumps(qry(f"PRAGMA table_info({table})"))


@mcp.tool()
def run_query(sql: str) -> str:
    """Execute a SQL SELECT query on the movies database. Supports JOINs across
    movies, directors, actors, cast_members, and reviews tables."""
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries allowed via run_query"})
    return json.dumps(qry(sql), default=str)


@mcp.tool()
def add_review(movie_title: str, reviewer_name: str, rating: int, comment: str) -> str:
    """Add a review for a movie. rating must be 1-10."""
    movies = qry("SELECT id, title FROM movies WHERE title LIKE ?", (f"%{movie_title}%",))
    if not movies:
        return json.dumps({"error": f"No movie found matching '{movie_title}'"})
    movie = movies[0]
    res = exe(
        "INSERT INTO reviews (movie_id, reviewer_name, rating, comment, review_date) VALUES (?,?,?,?,?)",
        (movie["id"], reviewer_name, max(1, min(10, rating)), comment, date.today().isoformat()),
    )
    return json.dumps({"success": True, "movie": movie["title"], "review_id": res["last_insert_id"]})


if __name__ == "__main__":
    mcp.run()
