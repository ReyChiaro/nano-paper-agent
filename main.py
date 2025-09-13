# paper_agent/main.py

import os
import sys

from utils.config import config
from utils.logger import logger
from database.db_manager import DBManager
from parsers.pdf_parser import PDFParser
from llm.llm_interface import LLMInterface
from embeddings.embedding_model import EmbeddingModel
from rag.retriever import Retriever
from rag.generator import Generator
from management.paper_manager import PaperManager
from ui.cli import CLI  # Import CLI


def main():
    """
    Main entry point for the Paper Agent application.
    Initializes core components and starts the CLI.
    """
    logger.info("Starting Paper Agent application...")

    # --- Configuration and Directory Setup ---
    papers_dir = config.get("PAPERS_DIR")
    db_dir = config.get("DB_DIR")

    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    logger.info(f"Papers directory: {papers_dir}")
    logger.info(f"Database directory: {db_dir}")
    logger.info(f"Database name: {config.get('DATABASE_NAME')}")

    # --- Initialize Core Components ---
    try:
        llm_interface = LLMInterface()
        embedding_model = EmbeddingModel()
        db_manager = DBManager()
        pdf_parser = PDFParser(llm_interface)
        retriever = Retriever(db_manager, embedding_model)
        generator = Generator(llm_interface)

        # --- Initialize PaperManager ---
        paper_manager = PaperManager(db_manager, pdf_parser, embedding_model, retriever, generator, llm_interface)
        logger.info("PaperManager initialized successfully.")

        # --- Start the CLI ---
        cli = CLI(paper_manager)
        cli.run()

    except Exception as e:
        logger.critical(f"An unrecoverable error occurred during application startup: {e}", exc_info=True)
        sys.exit("Paper Agent failed to start. Check logs for details.")


if __name__ == "__main__":
    main()
