import os
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader
from dotenv import load_dotenv

from rag import Chunk, RAGPipeline, VectorStore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_num, "text": text})

    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("CHUNK_SIZE must be greater than CHUNK_OVERLAP")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap

    return chunks


def build_chunks_for_pdf(pdf_path: Path) -> List[Dict]:
    pages = extract_pdf_pages(pdf_path)
    records = []
    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]
        page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(page_chunks, start=1):
            records.append(
                {
                    "text": chunk,
                    "metadata": {
                        "file_name": pdf_path.name,
                        "page": page_num,
                        "chunk_id": idx,
                    },
                }
            )
    return records


def ingest_all_pdfs() -> None:
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    pipeline = RAGPipeline()
    store = VectorStore()
    store.clear()

    all_records = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        all_records.extend(build_chunks_for_pdf(pdf_file))

    texts = [record["text"] for record in all_records]
    embeddings = pipeline.embed_texts(texts)

    chunks = []
    for record, embedding in zip(all_records, embeddings):
        chunks.append(
            Chunk(
                text=record["text"],
                metadata=record["metadata"],
                embedding=embedding,
            )
        )

    store.add_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(pdf_files)} PDF file(s).")


if __name__ == "__main__":
    ingest_all_pdfs()