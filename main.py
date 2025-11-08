"""
RAG Agent using LangChain + ChromaDB + HuggimgFace local models.
Workflow: plan → retrieve → answer → reflect
"""

import os
import re
import glob
import json
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ========== SETUP ==========

# Replace this with your API key
load_dotenv()

# --- Embeddings (local / free) ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- Local LLM via transformers ---
print("[INFO] Loading local FLAN-T5 model (this may take a bit the first time)...")
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

gen_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

llm = HuggingFacePipeline(pipeline=gen_pipeline)
print("[INFO] Local model ready.")

import re

def reframe_text(raw: str) -> str:
    """
    Clean and reframe raw stitched text so it reads naturally:
    - Fix section headings and numeric artifacts
    - Ensure proper sentence punctuation and spacing
    - Join broken lines smoothly
    """
    # Remove section numbers like "1." or "2."
    cleaned = re.sub(r"\b\d+\.\s*", "", raw)

    # Replace multiple newlines or spaces with single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Fix missing periods between sentences that run together
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1. \2", cleaned)

    # Add space after punctuation if missing
    cleaned = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", cleaned)

    # Trim leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned


def clean_context_text(raw: str) -> str:
    """
    Clean up raw retrieved text so the LLM sees something more coherent:
    - Remove empty lines and section-number-only headings (like '1. Introduction')
    - Collapse weird newlines and multiple spaces
    """
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # remove bare section heading lines like "1. Introduction", "4. Discussion", "5. Conclusion"
        if re.match(r"^\d+\.\s*(introduction|discussion|results|conclusion)\.?$", stripped.lower()):
            continue

        lines.append(stripped)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ========== LOAD DOCUMENTS ==========

def load_documents(data_path="data"):
    docs = []
    for path in glob.glob(f"{data_path}/*"):
        if path.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs


print("[INFO] Loading documents from 'data/' folder...")
raw_docs = load_documents("data")
print(f"[INFO] Loaded {len(raw_docs)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(raw_docs)
print(f"[INFO] Split into {len(docs)} chunks")
ALL_CHUNKS = docs
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="rag_collection",
)
print("[INFO] ChromaDB index created.")


# ========== NODES ==========

def plan_node(state: dict) -> dict:
    question = state["question"]
    print(f"\n[PLAN] Question: {question}")
    state["need_retrieval"] = True
    print("[PLAN] Retrieval needed: True")
    return state


def retrieve_node(state: dict) -> dict:
    if not state.get("need_retrieval", False):
        print("[RETRIEVE] Skipping retrieval")
        state["docs"] = []
        return state

    question = state["question"]
    print(f"[RETRIEVE] Searching in Chroma for: {question!r}")

    # 1) Normal semantic search
    retrieved = vectorstore.similarity_search(question, k=6)

    # 2) Extra: keyword/phrase-based boost from ALL_CHUNKS
    extra_chunks = []
    q_lower = question.lower()

    # build 3-word phrases (trigrams) from the question
    words = re.findall(r"[a-zA-Z]+", q_lower)
    phrases = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]

    for chunk in ALL_CHUNKS:
        text = chunk.page_content.lower()

        # if any 3-word phrase from the question appears in this chunk, keep it
        if any(p and p in text for p in phrases):
            extra_chunks.append(chunk)

    # 3) Combine and deduplicate by text
    combined = retrieved + extra_chunks
    unique = []
    seen = set()
    for d in combined:
        key = d.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(d)

    state["docs"] = unique
    print(f"[RETRIEVE] Retrieved {len(unique)} documents (including phrase matches)")
    for i, d in enumerate(unique[:6], start=1):
        print(f"\n[DOC {i}]\n{d.page_content[:300]}...\n")

    return state

def answer_node(state: dict) -> dict:
    question = state["question"].strip()
    docs = state.get("docs", [])
    context = "\n\n".join([d.page_content for d in docs])

    # Extract numbers (for targeted numeric questions)
    question_numbers = re.findall(r"\d+", question)
    sentences = re.split(r'(?<=[.!?])\s+', context)
    numeric_sentences = [s.strip() for s in sentences if any(ch.isdigit() for ch in s)]

    best_sentence = None
    best_distance = float("inf")

    # --- NUMERIC TARGETED LOGIC ---
    if question_numbers:
        qn = [int(n) for n in question_numbers]
        for s in numeric_sentences:
            nums_in_s = [int(n) for n in re.findall(r"\d+", s)]
            if not nums_in_s:
                continue

            if any(str(q) in s for q in question_numbers) or any(f"{q}+" in s for q in question_numbers):
                best_sentence = s
                best_distance = 0
                break

            distance = min(abs(sn - q) for sn in nums_in_s for q in qn)
            if distance < best_distance:
                best_distance = distance
                best_sentence = s

        if best_sentence:
            print(f"[ANSWER] Found closest numeric sentence: {best_sentence}")
            state["answer"] = best_sentence
            return state

    # --- BROAD QUESTION FALLBACK ---
    # Use the local LLM to summarize the top retrieved context
    # --- Definition-style questions ---
    if question.lower().startswith(("what is", "what is meant by")):
        system_prompt = (
        "You are a helpful assistant that explains the meaning of the given term clearly, "
        "based only on the provided context. If the context lists examples or effects "
        "instead of a formal definition, summarize those to infer what the term means. "
        "Write 2–3 sentences that define the concept in your own words."
        )



    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely using only the facts in context."
    
    print("[ANSWER] Generating summarized answer with LLM...")
    answer = llm.invoke(f"{system_prompt}\n\n{user_prompt}").strip()
    answer = reframe_text(answer)  # 🔹 new line to clean punctuation and spacing
    print(f"[ANSWER] {answer}")

    state["answer"] = answer
    return state





# Load a lightweight similarity model once (outside the function)
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def reflect_node(state: dict) -> dict:
    question = state["question"]
    answer = (state.get("answer") or "").strip()
    docs = state.get("docs", [])
    context = "\n\n".join([d.page_content for d in docs])

    print("[REFLECT] Evaluating semantic similarity between answer and retrieved context...")

    if not answer:
        evaluation = {
            "ok": False,
            "feedback": "No answer was generated."
        }
        state["evaluation"] = evaluation
        return state

    # Compute embeddings
    emb_answer = similarity_model.encode(answer, convert_to_tensor=True)
    emb_context = similarity_model.encode(context, convert_to_tensor=True)

    # Cosine similarity (semantic closeness)
    score = util.pytorch_cos_sim(emb_answer, emb_context).item()

    print(f"[REFLECT] Semantic similarity score: {score:.3f}")

    # Interpret score
    if score >= 0.70:
        feedback = "The answer is semantically consistent with the retrieved document."
        ok = True
    elif score >= 0.45:
        feedback = "The answer is somewhat related but might not be from the correct section."
        ok = False
    else:
        feedback = "The answer does not align with the retrieved context."
        ok = False

    evaluation = {
        "ok": ok,
        "feedback": feedback,
        "similarity_score": round(score, 3)
    }

    print(f"[REFLECT] ok = {ok}")
    print(f"[REFLECT] feedback = {feedback}")

    state["evaluation"] = evaluation
    return state



# ========== PIPELINE RUNNER ==========

def run_agent(question: str):
    state = {"question": question}
    state = plan_node(state)
    state = retrieve_node(state)
    state = answer_node(state)
    state = reflect_node(state)

    print("\n=== FINAL ANSWER ===")
    print(state["answer"])
    print("\n=== REFLECTION ===")
    print(state["evaluation"])
    return state


# ========== MAIN ==========
if __name__ == "__main__":
    q = input("Ask a question: ")
    run_agent(q)
