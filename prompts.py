"""Part 1 - three prompt templates (zero-shot, few-shot, role-based)."""

ZERO_SHOT = """Answer the question using the context below.

Context:
{context}

Question: {question}
Answer:"""

FEW_SHOT = """Answer the question using ONLY the context. Keep answers short and factual.
If the context does not contain the answer, say "I could not find this in the documents."

Example 1
Context: Chroma is an open-source vector database that stores embeddings and supports similarity search.
Question: What does Chroma store?
Answer: Chroma stores embeddings and lets you run similarity search over them.

Example 2
Context: Overfitting happens when a model memorises training data and performs poorly on new data.
Question: Why is overfitting a problem?
Answer: The model memorises its training data, so it performs poorly on new, unseen data.

Example 3
Context: Python was created by Guido van Rossum.
Question: Who invented the telephone?
Answer: I could not find this in the documents.

Now your turn.
Context:
{context}

Question: {question}
Answer:"""

ROLE_BASED = """You are a senior technical document analyst. You read documents carefully, \
answer only from the supplied context, cite the source file in square brackets, \
and structure answers as: a one-line direct answer, then 2-3 supporting points. \
If the context is insufficient, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Analyst response:"""

TEMPLATES = {"zero_shot": ZERO_SHOT, "few_shot": FEW_SHOT, "role_based": ROLE_BASED}

DEFAULT_QUESTIONS = [
    "What is the difference between semantic search and keyword search?",
    "Why do we split documents into chunks before embedding them?",
    "How does Retrieval-Augmented Generation reduce hallucinations?",
    "What is a vector database used for?",
    "What does cosine similarity measure?",
]

JUDGE_PROMPT = """You are an impartial evaluator. Score the answer to the question from 1 to 5 \
on accuracy (matches the context), clarity (easy to read), and relevance (answers what was asked).
Be strict: give 5 only for perfect answers. Penalise missing source citations, \
extra information not in the context, and unclear structure.
Return ONLY JSON like {{"accuracy": 4, "clarity": 5, "relevance": 4}}.

Context:
{context}

Question: {question}

Answer to score:
{answer}"""
