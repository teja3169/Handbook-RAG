import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store" / "index.json"


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]


class VectorStore:
    def __init__(self, path: Path = VECTOR_STORE_PATH):
        self.path = path
        self.chunks: List[Chunk] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.chunks = []
            return

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [Chunk(**item) for item in data]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(
                [chunk.__dict__ for chunk in self.chunks],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        self.save()

    def clear(self) -> None:
        self.chunks = []
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Chunk]:
        if not self.chunks:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        scored = []
        for chunk in self.chunks:
            emb = np.array(chunk.embedding, dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            denom = emb_norm * query_norm
            score = float(np.dot(emb, query) / denom) if denom != 0 else 0.0
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


class RAGPipeline:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

        self.client = OpenAI(api_key=api_key)
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.top_k = int(os.getenv("TOP_K", "5"))
        self.store = VectorStore()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def answer(self, question: str) -> Dict[str, Any]:
        # Reload store each time so newly ingested files are visible
        self.store = VectorStore()

        if not self.store.chunks:
            return {
                "answer": "No indexed PDFs found yet. Please upload a PDF and wait for indexing to finish.",
                "sources": [],
            }

        query_embedding = self.embed_query(question)
        retrieved = self.store.search(query_embedding, top_k=self.top_k)

        if not retrieved:
            return {
                "answer": "I could not find anything relevant in the indexed PDFs.",
                "sources": [],
            }

        context_blocks = []
        sources = []

        for idx, chunk in enumerate(retrieved, start=1):
            meta = chunk.metadata
            file_name = meta.get("file_name", "unknown")
            page = meta.get("page", "?")
            chunk_id = meta.get("chunk_id", "?")

            source_label = f"{file_name} - page {page}"
            context_blocks.append(f"[Source {idx}] {source_label}\n{chunk.text}")

            sources.append(
                {
                    "file_name": file_name,
                    "page": page,
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                }
            )

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a helpful assistant for PDF question answering. "
            "Answer only from the provided context. "
            "If the answer is not contained in the context, say you do not know. "
            "When possible, mention the source file and page number."
        )

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n\n{context_text}"
        )

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content or "No answer generated."
        return {"answer": answer, "sources": sources}

    def ask(self, question: str) -> str:
        result = self.answer(question)
        return result["answer"]