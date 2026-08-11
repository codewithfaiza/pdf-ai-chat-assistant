import os
import shutil
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # loads .env file into environment variables

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from groq import Groq
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    doc_id: str

# ---------------------------------------------------------------------------
# ChromaDB + Groq (lazy init on startup)
# ---------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path="chroma_db")
# ChromaDB uses its built-in default embedding function (all-MiniLM-L6-v2
# via onnxruntime) — no external API key needed for embeddings.
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

groq_client: Groq | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client
    try:
        groq_client = Groq()  # reads GROQ_API_KEY from env
    except Exception as e:
        print(f"Warning: Failed to initialize Groq client: {e}")
    yield


app = FastAPI(title="PDF Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_groq() -> Groq:
    global groq_client
    if groq_client is None:
        try:
            groq_client = Groq()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Groq Client initialization error: {str(e)}")
    return groq_client


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    return splitter.split_text(text)


def store_chunks(chunks: list[str], doc_id: str):
    """Store text chunks in ChromaDB — embedding is handled automatically
    by ChromaDB's built-in default embedding function."""
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=[{"doc_id": doc_id} for _ in chunks],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"status": "Backend is running", "app": "PDF Chat API"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from this PDF file (it may be scanned or image-only)."
            )

        chunks = chunk_text(full_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to divide document into readable text chunks.")

        store_chunks(chunks, doc_id=file.filename)

        return {
            "filename": file.filename,
            "char_count": len(full_text),
            "chunk_count": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    active_groq = get_groq()

    try:
        # 1. Search Chroma for the most relevant chunks
        #    ChromaDB embeds the query text automatically using its default function.
        results = collection.query(
            query_texts=[request.question],
            n_results=4,
            where={"doc_id": request.doc_id},
        )

        relevant_chunks = results.get("documents", [[]])[0] if results.get("documents") else []
        if not relevant_chunks:
            return {"answer": "No relevant context found in the uploaded document for this question."}

        context = "\n\n".join(relevant_chunks)

        # 2. Ask Groq (Llama 3.3 70B) to answer using only that context
        completion = active_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"Answer the question using ONLY the context below. "
                                f"If the answer isn't in the context, say you don't know.\n\n"
                                f"Context:\n{context}\n\nQuestion: {request.question}"
                }
            ]
        )

        return {"answer": completion.choices[0].message.content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")