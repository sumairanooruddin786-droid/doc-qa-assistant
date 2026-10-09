import io, json, os, re, uuid
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from prompts import DEFAULT_QUESTIONS, JUDGE_PROMPT, TEMPLATES

load_dotenv()
BASE = Path(__file__).parent
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_WORDS, OVERLAP_WORDS, TOP_K = 120, 30, 3

app = FastAPI(title="AI Knowledge Assistant")
embedder = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=str(BASE / "chroma_db"))
col = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

_llm = None
def llm():
    global _llm
    if _llm is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise HTTPException(500, "GROQ_API_KEY missing. Add it to your .env file.")
        from groq import Groq
        _llm = Groq(api_key=key)
    return _llm

def chat(prompt: str, temperature: float = 0.2) -> str:
    r = llm().chat.completions.create(
        model=LLM_MODEL, temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip()

# ---------- Part 2: load -> chunk -> embed -> store ----------
def extract_text(name: str, data: bytes) -> str:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if ext in ("txt", "md", "markdown"):
        return data.decode("utf-8", errors="ignore")
    raise HTTPException(400, f"Unsupported file type: .{ext}")

def chunk_text(text: str):
    words = re.sub(r"\s+", " ", text).split(" ")
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words), 1), step)
            if words[i:i + CHUNK_WORDS]]

def ingest(name: str, data: bytes) -> int:
    chunks = chunk_text(extract_text(name, data))
    if not chunks:
        raise HTTPException(400, "No readable text found in file.")
    col.delete(where={"source": name})  # re-upload replaces old version
    embs = embedder.encode(chunks, normalize_embeddings=True).tolist()
    col.add(ids=[f"{name}-{i}-{uuid.uuid4().hex[:6]}" for i in range(len(chunks))],
            documents=chunks, embeddings=embs,
            metadatas=[{"source": name, "chunk": i} for i in range(len(chunks))])
    return len(chunks)

def retrieve(question: str, k: int = TOP_K):
    if col.count() == 0:
        raise HTTPException(400, "No documents yet. Upload some first.")
    q = embedder.encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=min(k, col.count()))
    return [{"text": d, "source": m["source"], "chunk": m["chunk"],
             "distance": round(dist, 4), "similarity": round(1 - dist, 4)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

def context_of(hits):
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

@app.on_event("startup")
def load_samples():
    if col.count() == 0:
        for f in sorted((BASE / "sample_docs").glob("*")):
            ingest(f.name, f.read_bytes())

# ---------- API ----------
class Q(BaseModel):
    question: str
    technique: str = "role_based"

class CompareReq(BaseModel):
    questions: list[str] | None = None
    judge: bool = True

@app.get("/api/documents")
def documents():
    got = col.get(include=["metadatas"])
    counts = {}
    for m in got["metadatas"]:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"name": n, "chunks": c} for n, c in sorted(counts.items())]

@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        out.append({"name": f.filename, "chunks": ingest(f.filename, await f.read())})
    return out

@app.delete("/api/documents/{name}")
def delete_doc(name: str):
    col.delete(where={"source": name})
    return {"deleted": name}

@app.post("/api/search")
def search(q: Q):
    return {"query": q.question, "results": retrieve(q.question)}

@app.post("/api/ask")
def ask(q: Q):
    if q.technique not in TEMPLATES:
        raise HTTPException(400, "technique must be zero_shot, few_shot or role_based")
    hits = retrieve(q.question)
    answer = chat(TEMPLATES[q.technique].format(context=context_of(hits), question=q.question))
    return {"answer": answer, "technique": q.technique, "sources": hits}

@app.post("/api/compare")
def compare(req: CompareReq):
    questions = req.questions or DEFAULT_QUESTIONS
    rows = []
    for question in questions:
        hits = retrieve(question)
        ctx = context_of(hits)
        row = {"question": question, "results": {}}
        for name, tpl in TEMPLATES.items():
            ans = chat(tpl.format(context=ctx, question=question))
            score = None
            if req.judge:
                try:
                    raw = chat(JUDGE_PROMPT.format(context=ctx, question=question, answer=ans), 0)
                    score = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                except Exception:
                    score = None
            row["results"][name] = {"answer": ans, "score": score}
        rows.append(row)
    totals = {}
    for name in TEMPLATES:
        s = [r["results"][name]["score"] for r in rows if r["results"][name]["score"]]
        totals[name] = round(sum(sum(x.values()) / 3 for x in s) / len(s), 2) if s else None
    return {"rows": rows, "average_scores": totals}

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
