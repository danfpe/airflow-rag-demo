# Airflow + RAG Learning Demo

This project is intentionally small. Its purpose is to teach how Airflow orchestrates jobs.

## Architecture

Documents
  -> load_documents
  -> parse_documents
  -> chunk_documents
  -> create_embeddings
  -> store_vectors
  -> validate_index
  -> PostgreSQL + pgvector

The embedding step uses a tiny deterministic hashed embedding, so no OpenAI key is required.

## Prerequisites

- Docker Desktop
- Docker Compose
- At least ~4 GB available memory for Airflow/Docker

On Windows, use Docker Desktop with Linux containers.

## Start

From this directory:

```bash
docker compose up --build
```

Then open:

http://localhost:8080

This learning setup disables authentication through Airflow SimpleAuthManager.

## Run the DAG

In the Airflow UI:

1. Open `rag_indexing_pipeline`.
2. Click Trigger.
3. Open Graph view.
4. Watch the tasks become green from left to right.
5. Open each task and inspect Logs.
6. In `validate_index`, inspect the retrieval results.

Expected dependency graph:

```text
load_documents
      |
parse_documents
      |
chunk_documents
      |
create_embeddings
      |
store_vectors
      |
validate_index
```

## Inspect pgvector manually

```bash
docker compose exec postgres \
  psql -U airflow -d ragdb \
  -c "SELECT id, source, LEFT(content, 90) FROM rag_chunks ORDER BY id;"
```

Count rows:

```bash
docker compose exec postgres \
  psql -U airflow -d ragdb \
  -c "SELECT COUNT(*) FROM rag_chunks;"
```

## What Airflow is doing

The DAG file defines the workflow and dependencies.

The scheduler decides which task can run next.

The executor runs tasks that are ready.

PostgreSQL `airflow` database stores Airflow metadata.

PostgreSQL `ragdb` database stores our RAG vectors.

TaskFlow return values create XCom relationships. In this demo, the XCom values are mostly file paths and small
metadata. The actual document/chunk data is written to `/opt/airflow/data/work`, which is mounted to `./data/work`
on the host.

## Important production lesson

Do not put large documents, large DataFrames, or embedding matrices directly in XCom.

Prefer:

Task A -> S3/path/table -> Task B

and pass only the path, object key, batch id, table name, or other small metadata through XCom.

## Reset everything

```bash
docker compose down -v
```

Then:

```bash
docker compose up --build
```

## Next upgrade

Replace `hashed_embedding()` with a real embedding model, then add a separate FastAPI online query service:

User -> FastAPI -> embed question -> pgvector retrieval -> LLM -> answer

Airflow remains responsible for the offline indexing pipeline.
