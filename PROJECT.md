```sh
paper_agent/
├── data/
│   ├── papers/             # Stores PDF papers
│   └── db/                 # Stores SQLite database
├── embeddings/             # Handles embedding generation
│   ├── __init__.py
│   └── embedding_model.py
├── parsers/                # Handles PDF parsing
│   ├── __init__.py
│   └── pdf_parser.py
├── rag/                    # Handles RAG logic
│   ├── __init__.py
│   └── retriever.py
│   └── generator.py
├── llm/                    # Handles LLM/VLM interactions
│   ├── __init__.py
│   └── llm_interface.py
├── database/               # Handles database operations
│   ├── __init__.py
│   └── db_manager.py
│   └── schema.sql          # SQL schema for database
├── management/             # Handles paper management (tags, status)
│   ├── __init__.py
│   └── paper_manager.py
├── ui/                     # Handles user interface (terminal UI)
│   ├── __init__.py
│   └── cli.py
├── utils/                  # Utility functions (logging, config)
│   ├── __init__.py
│   └── config.py
│   └── logger.py
├── main.py                 # Main entry point for the application
└── README.md               # Project documentation
```