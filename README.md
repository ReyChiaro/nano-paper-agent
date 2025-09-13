# Paper Agent

> A fully AI-generated PaperAgent, including the basic functions of RAG system like indexing, embedding, query and a simple interaction CLI UI. The LLM/VLMs used in this project can be replaced by all off-the-shelf models that support `OpenAI` API interfaces.
> 
> WARNING: This project only deployed locally, do not collect any information about the user.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge)

## 📚 Your Personal AI-Powered Academic Paper Assistant

Paper Agent is a local-first, AI-driven tool designed to help you manage, understand, and query your academic papers. It leverages Large Language Models (LLMs) and embedding models to extract metadata, chunk content, generate embeddings, summarize papers, and answer questions based on your personal research library.

---

## ✨ Features

*   **Intelligent PDF Ingestion:**
    *   Automatically extracts metadata (Title, Authors, Abstract) from PDF papers using an LLM, even with varied layouts.
    *   Chunks paper content into manageable sections for effective retrieval.
    *   Generates semantic embeddings for each section using a pre-trained Sentence Transformer model.
*   **Local Database Storage:**
    *   Stores paper metadata, sections, and embeddings in a local SQLite database.
    *   Supports tags and references for better organization.
*   **Retrieval-Augmented Generation (RAG):**
    *   Ask natural language questions about your entire paper library.
    *   The agent intelligently retrieves the most relevant sections from your papers.
    *   An LLM synthesizes an answer based *only* on the retrieved context, minimizing hallucination.
*   **Paper Summarization:**
    *   Generate concise, comprehensive LLM-powered summaries of individual papers.
*   **Command Line Interface (CLI):**
    *   Interact with your paper library directly from the terminal.
    *   Commands for adding, listing, detailing, querying, summarizing, tagging, and deleting papers.
*   **Configurable LLM & Embedding Models:**
    *   Easily switch between different LLM providers (e.g., OpenAI, Anthropic, or local LLMs via `LLMInterface` extension).
    *   Uses `sentence-transformers` for embeddings, with configurable models.

---

## 🚀 Getting Started

Follow these steps to set up and run your Paper Agent.

### Prerequisites

*   Python 3.11+
*   `uv`

### 1. Clone the Repository

```bash
git clone https://github.com/ReyChiaro/nano-paper-agent.git
cd nano-paper-agent
```

### 2. Install Dependencies

This project use `uv` to manage environment.

```bash
uv sync
```

### 3. Configure Your Agent

A `config.json` file is used to set up paths and API keys.
Create a `config.json` file in the root directory of the project (e.g., `paper_agent/config.json`) with the following structure:

```json
{
  "PAPERS_DIR": "data/papers",
  "DB_DIR": "data/db",
  "DATABASE_NAME": "paper_agent.db",
  "LOG_FILE": "logs/paper_agent.log",
  "LOG_LEVEL": "INFO",
  "LLM_API_KEY": "YOUR_OPENAI_API_KEY_OR_OTHER_LLM_KEY",
  "LLM_MODEL_NAME": "gpt-3.5-turbo",
  "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
  "PDF_CHUNK_SIZE": 1000,
  "PDF_CHUNK_OVERLAP": 200,
  "LLM_MAX_CONTEXT_CHARS_FOR_SUMMARY": 10000,
  "LLM_MAX_TOKENS_FOR_SUMMARY": 750
}
```

**Important Notes on Configuration:**

*   **`LLM_API_KEY`**:
    *   If you want to use a real LLM (like OpenAI's GPT models), replace `"YOUR_OPENAI_API_KEY_OR_OTHER_LLM_KEY"` with your actual API key.
    *   If you leave it as the placeholder, the `LLMInterface` will **simulate** LLM responses for metadata extraction, summarization, and RAG answers. This allows you to test the system without an API key or incurring costs.
*   **`LLM_MODEL_NAME`**: Specify the LLM model you wish to use (e.g., `"gpt-4-turbo"`, `"claude-3-opus-20240229"`). If `LLM_API_KEY` is a placeholder, this value is ignored.
*   **`EMBEDDING_MODEL_NAME`**: The default `sentence-transformers/all-MiniLM-L6-v2` is a good balance of performance and size. It will be downloaded automatically on first use.
*   **Paths**: `PAPERS_DIR` and `DB_DIR` are relative to the project root. The agent will create these directories if they don't exist.

### 4. Prepare Your Paper Library

Place your academic PDF files into the directory specified by `PAPERS_DIR` in your `config.json` (e.g., `paper_agent/data/papers/`).

### 5. Run the Paper Agent CLI

```bash
python main.py
```

You will be greeted by the interactive command-line interface:

```
--- Welcome to Paper Agent CLI ---
Type 'help' for a list of commands.
Type 'exit' to quit.

PaperAgent>
```

---

## 📖 CLI Usage

Here's a list of commands you can use in the Paper Agent CLI:

*   **`help`**: Display a list of all available commands and their usage.
*   **`add <path_to_pdf>`**:
    *   Adds a new paper to your library from a PDF file.
    *   Example: `add data/papers/my_research_paper.pdf`
    *   The agent will extract metadata, chunk content, generate embeddings, and store it.
*   **`list`**:
    *   Lists all papers currently in your library with their basic information.
*   **`details <paper_id>`**:
    *   Shows comprehensive details for a specific paper, including abstract, file path, tags, and references.
    *   Example: `details 1`
*   **`query <your_question>`**:
    *   Ask a natural language question about the content of your papers. The RAG system will retrieve relevant information and generate an answer.
    *   Example: `query What are the key findings of the paper on transformer models?`
*   **`summarize <paper_id>`**:
    *   Generates a detailed summary of a specific paper using the LLM. The summary is then stored with the paper's metadata.
    *   Example: `summarize 2`
*   **`tag <paper_id> <tag_name>`**:
    *   Adds a tag (e.g., "AI", "NLP", "Review") to a paper for better organization.
    *   Example: `tag 1 Machine Learning`
*   **`untag <paper_id> <tag_name>`**:
    *   Removes an existing tag from a paper.
    *   Example: `untag 1 Machine Learning`
*   **`delete <paper_id>`**:
    *   Permanently deletes a paper and all its associated data (sections, embeddings, tags) from your library.
    *   **Caution: This action is irreversible.**
    *   Example: `delete 3`
*   **`exit`**:
    *   Exits the Paper Agent CLI.

---

## 🛠️ Project Structure

```
paper_agent/
├── data/
│   ├── db/                 # SQLite database and related files
│   └── papers/             # Your PDF papers are stored here
├── database/
│   ├── __init__.py
│   └── db_manager.py       # Handles all database interactions
├── embeddings/
│   ├── __init__.py
│   └── embedding_model.py  # Loads and uses the Sentence Transformer model
├── llm/
│   ├── __init__.py
│   └── llm_interface.py    # Abstraction for LLM API calls (e.g., OpenAI, simulation)
├── management/
│   ├── __init__.py
│   └── paper_manager.py    # Orchestrates core paper management logic
├── parsers/
│   ├── __init__.py
│   └── pdf_parser.py       # Extracts text from PDFs, LLM-enhanced metadata extraction
├── rag/
│   ├── __init__.py
│   ├── generator.py        # Uses LLM to generate answers from retrieved context
│   └── retriever.py        # Finds relevant sections using embeddings
├── ui/
│   ├── __init__.py
│   └── cli.py              # Command Line Interface for user interaction
├── utils/
│   ├── __init__.py
│   ├── config.py           # Loads and manages configuration
│   └── logger.py           # Centralized logging setup
├── config.json             # Configuration file
├── main.py                 # Main application entry point
└── pyproject.toml          # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

*   [pypdf](https://pypdf.readthedocs.io/en/stable/) for PDF parsing.
*   [sentence-transformers](https://www.sbert.net/) for efficient embeddings.
*   [SQLite](https://www.sqlite.org/index.html) for local data storage.
*   [OpenAI](https://openai.com/) for powerful language models.
*   [Qwen](https://github.com/QwenLM) for powerful language models.