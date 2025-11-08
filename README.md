# ☕ Quantum-Caffeinated Chaos  
**Retrieval-Augmented Generation (RAG) System with LangChain + ChromaDB + Hugging Face**

---

## 🎯 Overview  
**Quantum-Caffeinated Chaos** is a document-grounded **Retrieval-Augmented Generation (RAG)** system designed to retrieve, reason, and reflect using local large language models (LLMs).  
Built as a late-night experiment (and powered entirely by caffeine ☕), it demonstrates how local models can perform contextual reasoning and self-reflection without cloud APIs.

The agent uses a custom pipeline:
> **plan → retrieve → answer → reflect**  
allowing transparent reasoning and verification of each generated response.

---

## 🔧 Architecture & Tech Stack  

| Component | Description |
|------------|-------------|
| **LangChain** | Core RAG orchestration and document management |
| **ChromaDB** | Local vector database for semantic document retrieval |
| **Hugging Face Transformers** | Local `FLAN-T5` model for offline text generation |
| **Sentence-Transformers** | Embeddings using `all-MiniLM-L6-v2` for similarity search |
| **Streamlit** | Interactive interface for querying and result visualization |
| **Python** | Modular and extensible workflow with reflection nodes |

---

## 🧩 Key Features  
- Fully **offline RAG pipeline** – no cloud dependencies  
- **Semantic reflection** to validate grounding of answers in retrieved documents  
- **PDF and text ingestion** through LangChain document loaders  
- Dynamic numeric-matching logic for quantitative questions  
- Context-aware reasoning with transparency at every step  

---

## 🚀 Setup & Usage  

Clone the repository and set up your environment:

```bash
git clone https://github.com/reh-31/quantum-caffeinated-chaos.git
cd quantum-caffeinated-chaos
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

### 🔐 Environment Configuration

This project uses a `.env` file to store private configuration variables, such as your Hugging Face API key.  
Since `.env` files are excluded from version control for security, you’ll need to **create your own** before running the project.

Create a new file in the project root named `.env` and add:

```bash
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here

