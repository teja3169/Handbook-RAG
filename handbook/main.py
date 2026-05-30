import os
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from ingest import ingest_all_pdfs
from rag import RAGPipeline

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"

DATA_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Minimal RAG App")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

rag_pipeline = RAGPipeline()

ingestion_lock = threading.Lock()
ingestion_status = {
    "running": False,
    "last_message": "No indexing started yet.",
}


def run_ingestion_job() -> None:
    global ingestion_status

    if ingestion_lock.locked():
        return

    with ingestion_lock:
        ingestion_status["running"] = True
        ingestion_status["last_message"] = "Indexing started..."

        try:
            ingest_all_pdfs()
            ingestion_status["last_message"] = "Indexing completed successfully."
        except Exception as e:
            ingestion_status["last_message"] = f"Indexing failed: {str(e)}"
        finally:
            ingestion_status["running"] = False


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/upload", status_code=302)


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "message": None,
            "status": ingestion_status,
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    saved_files = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        file_path = DATA_DIR / file.filename
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        saved_files.append(file.filename)

    if saved_files:
        background_tasks.add_task(run_ingestion_job)
        message = f"Uploaded {len(saved_files)} PDF file(s). Indexing started automatically."
    else:
        message = "No valid PDF files were uploaded."

    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "message": message,
            "status": ingestion_status,
        },
    )


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "status": ingestion_status,
        },
    )


@app.get("/api/status")
def get_status():
    return JSONResponse(ingestion_status)


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()

    if not question:
        return JSONResponse({"error": "Question is required."}, status_code=400)

    if ingestion_status["running"]:
        return JSONResponse(
            {"answer": "Documents are still being indexed. Please wait a moment and try again."},
            status_code=200,
        )

    try:
        answer = rag_pipeline.ask(question)
        return JSONResponse({"answer": answer})
    except FileNotFoundError:
        return JSONResponse(
            {"answer": "No indexed documents found yet. Please upload a PDF first."},
            status_code=200,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)