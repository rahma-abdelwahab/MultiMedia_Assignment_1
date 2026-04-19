# 🚀 Vision-RAG: Multi-Modal Document Intelligence

**DSAI 413 - Assignment 1** A next-generation Retrieval-Augmented Generation (RAG) pipeline that bypasses traditional OCR limitations. By treating complex scientific documents (like bioinformatics papers) as visual patches, this system enables high-accuracy question answering across dense text, complex tables, and 3D figures.

-----

## 🌟 Core Capabilities

  * **OCR-Free Visual Retrieval:** Utilizes the Vision-Language Model **ColPali** to encode full pages and cropped figures directly from their visual layout.
  * **Smart Figure & Table Extraction:** Uses `PyMuPDF` to intelligently cluster vector drawings into padded figure crops, and `pdfplumber` to linearize table structures.
  * **Late-Interaction Scoring (MaxSim):** Performs token-to-patch similarity matching, allowing hyper-specific queries to find exact data points within charts or tables.
  * **Grounded Answer Synthesis:** Powered by **Llama 3.3 70B** (via Groq API) to generate concise answers with strict, inline source citations.
  * **Built-in Evaluation:** Features an automated test suite to calculate `Recall@K` across different document modalities.

-----

## 🏗️ System Architecture

1.  **Ingestion Module:** Downloads domain-specific PDFs from arXiv (`q-bio.BM`). Pages are rendered at 300 DPI, and distinct modalities (Text, Table, Figure) are isolated and cached with rich metadata.
2.  **Embedding & Indexing:** ColPali generates multi-vector embeddings for every document chunk. These are indexed into a local, on-disk **Qdrant** database.
3.  **Retrieval Engine:** Combines Approximate Nearest Neighbor (ANN) search with MaxSim late-interaction re-ranking to fetch the top $K$ most structurally and semantically relevant chunks.
4.  **Interactive Application:** A sleek Streamlit interface where users can query the knowledge base and inspect the underlying evidence (cropped images and parsed text) via expandable UI cards.

-----

## ⚙️ Quick Start Guide

> **Note:** This project relies on a local Qdrant instance and the Groq API. Ensure you have your API key ready.

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd ASSIGNMENT_1_2
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure Environment**
Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_api_key_here
```

**4. Build the Knowledge Base**
Run the ingestion pipeline to download, parse, and embed the dataset:

```bash
python data/download_papers.py
python ingestion/pdf_to_images.py
python ingestion/extract_text_tables.py
python ingestion/extract_figures.py
python ingestion/embed_pages.py
```

**5. Start the Application**

```bash
streamlit run app/streamlit_app.py
```

-----

## 📂 Repository Layout

```text
.
├── app/
│   └── streamlit_app.py          # Main chat interface and evaluation dashboard
├── data/                         # Local storage for all ingested artifacts
│   ├── figure_images/            # High-res cropped charts and figures
│   ├── page_images/              # 300 DPI full-page renders
│   ├── pdfs/                     # Raw arXiv PDF files
│   ├── qdrant_db/                # Qdrant local vector database
│   ├── embeddings.pt             # Cached tensor embeddings
│   └── *_metadata.json           # JSON trackers for modalities
├── evaluation/
│   └── benchmark.py              # Automated Recall@K test suite
├── generation/
│   └── qa_chain.py               # LLM prompt construction and Groq API logic
├── ingestion/
│   ├── embed_pages.py            # ColPali embedding generator
│   ├── extract_figures.py        # PyMuPDF path clustering
│   ├── extract_text_tables.py    # Table linearization logic
│   └── pdf_to_images.py          # PDF rendering script
└── retrieval/
    ├── retriever.py              # ANN + MaxSim retrieval logic
    └── vector_store.py           # Qdrant collection builder
```

-----

## 📊 Evaluation & Metrics

The system includes a benchmarking suite specifically designed for multi-modal evaluation. It tests the pipeline against queries requiring distinct visual or textual reasoning (e.g., *Text*, *Table*, *Figure*).

To run the benchmark and view the `Recall@K` metrics, simply navigate to the **Evaluation Suite** tab within the Streamlit application.
