# SKILL.md — Movie Database Agent

## Name
`movie-db-agent`

## Description
An AI agent that explores and queries a local movie database using natural language. Users can look up films, directors, actors, ratings, and reviews — or submit their own review — without writing any SQL.

## Instructions

1. Receive a natural language question or command from the user via the chat interface.
2. ALWAYS call `describe_table` before writing any query — never assume column names.
3. Call `run_query` with a valid SELECT statement to fetch results. Use JOINs for cross-table questions (e.g. actor + movie, director + rating).
4. For complex questions, break them into multiple sequential queries — do not guess at data.
5. If the user wants to submit a review, call `add_review` with the movie title, reviewer name, a rating (1–10), and a comment.
6. Format all results as JSON when the user asks for JSON, otherwise use markdown tables.
7. Never invent data — if a query returns nothing, say so and suggest a follow-up.

## Tools

| Tool | Type | Description |
|------|------|-------------|
| `list_tables` | SQLite | Returns all table names in `movies.db` |
| `describe_table` | SQLite | Returns column names and types for a given table |
| `run_query` | SQLite | Executes a SELECT query; JOINs across all tables supported |
| `add_review` | SQLite | Inserts a review row (movie_id resolved by fuzzy title match) |

### Database Schema
```
movies       (id, title, year, genre, rating, runtime_min, director_id)
directors    (id, name, nationality, birth_year)
actors       (id, name, nationality, birth_year)
cast_members (movie_id, actor_id, character_name)
reviews      (id, movie_id, reviewer_name, rating, comment, review_date)
```

## Dependencies

- **Runtime**: Python 3.10+
- **LLM**: `llama3.2:3b` via [Ollama](https://ollama.com) at `http://localhost:11434`
- **Frameworks**: LangGraph, LangChain-Ollama, Flask
- **Database**: SQLite (`movies.db`, seeded by `seed_db.py`)

## Authentication

No external authentication required. All resources are local:
- Ollama runs on `localhost:11434`
- Database is a local SQLite file (`movies.db`)

Set environment variables to override defaults:
```
MODEL_NAME=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

## Version

`1.0.0` — Initial release. Supports read queries, cross-table JOINs, and review submission.
