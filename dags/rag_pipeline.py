from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import timedelta
from pathlib import Path

import pendulum
import psycopg2

from airflow.sdk import dag, task


DOCS_DIR = Path("/opt/airflow/data/docs")
WORK_DIR = Path("/opt/airflow/data/work")
RAG_DB_DSN = os.environ.get(
    "RAG_DB_DSN",
    "postgresql://airflow:airflow@postgres:5432/ragdb",
)
EMBEDDING_DIM = 64


def hashed_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Small deterministic demo embedding.

    This is NOT a semantic production embedding model.
    It is a normalized hashed bag-of-words vector so the demo can run
    without any external API key.
    """
    vector = [0.0] * dim
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm:
        vector = [v / norm for v in vector]
    return vector


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


@dag(
    dag_id="rag_indexing_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["rag", "learning", "pgvector"],
)
def rag_indexing_pipeline():

    @task
    def load_documents() -> list[str]:
        """
        Job 1:
        Discover input documents.

        The returned list is small metadata, so passing it through XCom is OK.
        """
        paths = sorted(str(p) for p in DOCS_DIR.glob("*.txt"))
        if not paths:
            raise RuntimeError(f"No .txt files found under {DOCS_DIR}")

        print(f"Discovered {len(paths)} documents:")
        for path in paths:
            print(f" - {path}")

        return paths

    @task
    def parse_documents(document_paths: list[str]) -> str:
        """
        Job 2:
        Read raw documents and write parsed output to shared storage.

        Notice that we return a FILE PATH through XCom, not the full dataset.
        """
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        output_path = WORK_DIR / "parsed.jsonl"

        with output_path.open("w", encoding="utf-8") as out:
            for raw_path in document_paths:
                path = Path(raw_path)
                content = path.read_text(encoding="utf-8").strip()
                record = {
                    "source": path.name,
                    "content": content,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Parsed documents written to {output_path}")
        return str(output_path)

    @task
    def chunk_documents(parsed_path: str) -> str:
        """
        Job 3:
        Split parsed documents into overlapping chunks.
        """
        chunk_size = 420
        overlap = 80
        output_path = WORK_DIR / "chunks.jsonl"

        def split_text(text: str) -> list[str]:
            if len(text) <= chunk_size:
                return [text]

            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(text):
                    break
                start = max(0, end - overlap)
            return chunks

        chunk_count = 0
        with open(parsed_path, "r", encoding="utf-8") as src, \
             output_path.open("w", encoding="utf-8") as out:
            for line in src:
                record = json.loads(line)
                for index, content in enumerate(split_text(record["content"])):
                    chunk_id = f'{record["source"]}::{index}'
                    out.write(
                        json.dumps(
                            {
                                "id": chunk_id,
                                "source": record["source"],
                                "content": content,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    chunk_count += 1

        print(f"Created {chunk_count} chunks at {output_path}")
        return str(output_path)

    @task(
        retries=2,
        retry_delay=timedelta(seconds=10),
    )
    def create_embeddings(chunks_path: str) -> str:
        """
        Job 4:
        Create embeddings.

        Later, replace hashed_embedding() with OpenAI / Azure OpenAI /
        Hugging Face / local embedding model.
        """
        output_path = WORK_DIR / "embedded_chunks.jsonl"
        count = 0

        with open(chunks_path, "r", encoding="utf-8") as src, \
             output_path.open("w", encoding="utf-8") as out:
            for line in src:
                record = json.loads(line)
                record["embedding"] = hashed_embedding(record["content"])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"Created {count} embeddings at {output_path}")
        return str(output_path)

    @task
    def store_vectors(embedded_path: str) -> dict:
        """
        Job 5:
        Write vectors into PostgreSQL + pgvector.
        """
        conn = psycopg2.connect(RAG_DB_DSN)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS rag_chunks (
                            id TEXT PRIMARY KEY,
                            source TEXT NOT NULL,
                            content TEXT NOT NULL,
                            embedding VECTOR({EMBEDDING_DIM}) NOT NULL
                        )
                        """
                    )

                    # For a learning demo we rebuild the index every run.
                    # In production we would normally do incremental upserts.
                    cur.execute("TRUNCATE TABLE rag_chunks")

                    inserted = 0
                    with open(embedded_path, "r", encoding="utf-8") as src:
                        for line in src:
                            record = json.loads(line)
                            cur.execute(
                                """
                                INSERT INTO rag_chunks(id, source, content, embedding)
                                VALUES (%s, %s, %s, %s::vector)
                                ON CONFLICT (id) DO UPDATE SET
                                    source = EXCLUDED.source,
                                    content = EXCLUDED.content,
                                    embedding = EXCLUDED.embedding
                                """,
                                (
                                    record["id"],
                                    record["source"],
                                    record["content"],
                                    vector_literal(record["embedding"]),
                                ),
                            )
                            inserted += 1

            print(f"Stored {inserted} vectors in ragdb.rag_chunks")
            return {
                "table": "rag_chunks",
                "inserted": inserted,
            }
        finally:
            conn.close()

    @task
    def validate_index(store_result: dict) -> dict:
        """
        Job 6:
        Run one retrieval query to prove the vector index works.
        """
        question = "How does Airflow orchestrate jobs?"
        query_vector = vector_literal(hashed_embedding(question))

        conn = psycopg2.connect(RAG_DB_DSN)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        source,
                        content,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM rag_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT 3
                    """,
                    (query_vector, query_vector),
                )
                rows = cur.fetchall()

            print("=" * 70)
            print(f"QUESTION: {question}")
            print(f"INDEX ROWS INSERTED: {store_result['inserted']}")
            print("TOP RETRIEVED CHUNKS:")
            for rank, (source, content, similarity) in enumerate(rows, start=1):
                print(
                    f"\n#{rank} source={source} "
                    f"similarity={float(similarity):.4f}\n{content}"
                )
            print("=" * 70)

            return {
                "question": question,
                "top_sources": [row[0] for row in rows],
            }
        finally:
            conn.close()

    docs = load_documents()
    parsed = parse_documents(docs)
    chunks = chunk_documents(parsed)
    embeddings = create_embeddings(chunks)
    stored = store_vectors(embeddings)
    validate_index(stored)


rag_indexing_pipeline()
