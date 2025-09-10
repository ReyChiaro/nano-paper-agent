# Paper Agent

A personal AI assistant for managing, summarizing, querying, and recommending academic papers.

## Project Vision

This project aims to address common pain points for AI researchers by providing an intelligent agent that can:
1.  **Read and Summarize**: Automatically process PDF papers and generate concise summaries with solid evidence.
2.  **Management**: Organize papers locally using tags and metadata.
3.  **Query and Answer**: Engage in conversational Q&A based on the content of read papers for deeper comprehension.
4.  **Paper Recommendation**: Suggest new papers based on reading history and citation networks.

## Current Stage: Step 1 - Initial Setup and Basic Utilities

This initial phase sets up the foundational elements of the project:
*   **Project Structure**: Defined a clear, modular directory layout.
*   **Configuration Manager (`utils/config.py`)**: A centralized system to manage application settings (data paths, API keys, model names, etc.) loaded from `config.json`. It automatically creates a default `config.json` if one doesn't exist.
*   **Logging Utility (`utils/logger.py`)**: A consistent way to log application events, errors, and debugging information to both console and a log file, configurable via `config.json`.
*   **Main Entry Point (`main.py`)**: The primary script to run the application, responsible for initializing core components and ensuring necessary directories exist.

## Getting Started (Step 1)

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # (Once it's a git repo)
    cd paper_agent
