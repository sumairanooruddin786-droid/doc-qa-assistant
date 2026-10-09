# AI Knowledge Assistant (Document Q&A with RAG)

Upload PDF/TXT/MD files, search them by meaning, ask questions, and compare three prompting techniques.

## Run
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # then put your Groq key in .env (free: console.groq.com)
uvicorn app:app --reload
```
Open http://127.0.0.1:8000. The 5 files in `sample_docs/` load automatically on first start.

## Structure
- `app.py`: loading, chunking, embeddings (all-MiniLM-L6-v2), ChromaDB, retrieval, RAG endpoints
- `prompts.py`: zero-shot, few-shot, role-based templates + the 5 test questions
- `static/index.html`: frontend (Ask, Semantic search, Compare prompts)

## Part 2 explanation: semantic vs keyword search
Keyword search matches the exact words in the query, so it fails on synonyms and paraphrases ("forgot my login" will not find "reset your password"). Semantic search converts text into embeddings (vectors that capture meaning) and returns the chunks whose vectors are closest to the question vector (cosine distance), so it finds relevant passages even with no shared words.

## Part 1 report (fill in after you run Compare prompts)
Run the Compare tab with the 5 default questions, then copy your averages and observations here.
| Technique | Accuracy | Clarity | Relevance | Notes |
|---|---|---|---|---|
| Zero-shot | | | | Fast, but format varies and it may add outside knowledge |
| Few-shot | | | | Consistent short style, refuses when context lacks the answer |
| Role-based | | | | Structured answer with source citations, most professional |

Expected conclusion (verify with your own results): role-based usually wins on clarity and relevance because the persona fixes structure, tone and citation behaviour; few-shot is most consistent in format; zero-shot is simplest but least controlled.
