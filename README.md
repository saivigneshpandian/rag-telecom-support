# 📡 Multi-Source RAG — Telecom Support Assistant

A Retrieval-Augmented Generation system that answers customer-support questions by retrieving across **three different data sources at once** — an FAQ sheet, a database of resolved support tickets, and a PDF product guide — and grounding every answer strictly in what it retrieves.

The interesting part isn't "RAG over a PDF." It's that **different data types need different handling**, and this pipeline treats each one correctly.

---

## Why multi-source

Real support knowledge doesn't live in one place. It's spread across FAQs, past tickets, and manuals — and each has a different shape:

| Source | Format | How it's handled |
|---|---|---|
| **FAQ** | CSV rows | No chunking — one row = one document (splitting would break Q&A pairs) |
| **Tickets** | SQLite DB | No chunking — one resolved ticket = one document (keeps the full case intact) |
| **Guides** | PDF | Chunked with `RecursiveCharacterTextSplitter` (long-form prose needs splitting) |

Each source is embedded into its **own Chroma collection**, and a merged retriever queries all three, so an answer can draw on a policy from the FAQ, a real resolution from a past ticket, and a step from the guide — in a single response.

Retrieved documents are **tagged by source** (`[FAQ]`, `[TICKETS]`, `[GUIDES]`) before they reach the model, so the LLM knows where each piece of context came from.

---

## How it works

```
                 +--- faq.csv -----------> embed ->  Chroma: faq
Ingestion  ------+--- tickets.db --------> embed ->  Chroma: tickets
                 +--- telecom_guide.pdf -> chunk -> embed -> Chroma: guides
                                                       |
Query --> merged retriever (queries all 3) --> tag by source --> prompt
                                                       |
                                              Gemini 2.5 Flash
                                                       |
                                          grounded answer (or "call 611")
```

- **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace, runs locally — no embedding API cost)
- **Vector store:** Chroma, three named collections in one persistent store
- **LLM:** Google Gemini 2.5 Flash, `temperature=0` for consistent factual answers
- **Frontend:** Streamlit chat UI with sample questions

---

## Grounding

The system prompt constrains the model to answer **only from retrieved context**. If the context doesn't confidently cover the question, it says so and points the customer to call 611 or use the app — rather than inventing an answer. Hallucination in a support context means telling a customer something false about their bill or service, so this guardrail is the point, not a nicety.

---

## Tech stack

`Python` · `LangChain` · `Chroma` · `HuggingFace embeddings` · `Google Gemini 2.5 Flash` · `Streamlit` · `pandas` · `SQLite` · `PyMuPDF`

---

## Project structure

```
ingestion.ipynb         # CSV  (faq.csv)      -> faq collection
ingestion_pdf.ipynb     # PDF  (guide)        -> guides collection (chunked)
ingestion_ticket.ipynb  # SQLite (tickets.db) -> tickets collection
retriever.py            # merged retriever across all 3 collections
rag_chain.py            # retriever -> prompt -> Gemini -> answer
app.py                  # Streamlit chat frontend
```

---

## Setup

```bash
# 1. Install
pip install langchain langchain-chroma langchain-huggingface \
            langchain-google-genai langchain-community \
            streamlit pandas pymupdf python-dotenv

# 2. Add your key to .env
GOOGLE_API_KEY=your_key

# 3. Ingest each source (run the three notebooks once to build the vector store)
#    ingestion.ipynb  ->  ingestion_pdf.ipynb  ->  ingestion_ticket.ipynb

# 4. Run the app
streamlit run app.py
```

---

## Design notes

A few decisions worth calling out:

- **Chunking is not one-size-fits-all.** Splitting a FAQ row or a resolved ticket would sever the question from its answer. Only the long-form PDF gets chunked. Matching the strategy to the data shape is most of what makes retrieval quality good here.
- **Separate collections, merged at query time.** Keeping sources in distinct collections (rather than one big pile) means retrieval depth is tunable per source (`k_faq`, `k_tickets`, `k_guides`) and the model always knows provenance.
- **Local embeddings.** `all-MiniLM-L6-v2` runs on-device, so ingestion has no per-token embedding cost — only the final answer calls an API.

---

## Roadmap

- [ ] Add reranking across the merged results (currently concatenated by source)
- [ ] Cite the specific source doc(s) behind each answer in the UI
- [ ] Add conversation memory for multi-turn support sessions
- [ ] Swap to LangGraph for conditional retrieval (e.g., skip tickets for policy questions)
