# paper_agent/main.py

import os
from utils.config import config
from utils.logger import logger
from database.db_manager import DBManager
from parsers.pdf_parser import PDFParser
from llm.llm_interface import LLMInterface
from embeddings.embedding_model import EmbeddingModel
from rag.retriever import Retriever  # Import Retriever
from rag.generator import Generator  # Import Generator
import numpy as np
import sys


def main():
    """
    Main entry point for the Paper Agent application.
    Initializes core components and starts the UI.
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

    # --- Initialize LLMInterface ---
    llm_interface = LLMInterface()
    logger.info("LLMInterface initialized.")

    # --- Initialize EmbeddingModel ---
    try:
        embedding_model = EmbeddingModel()
        logger.info("EmbeddingModel initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize EmbeddingModel: {e}", exc_info=True)
        sys.exit("Application startup failed due to embedding model error.")

    # --- Initialize DBManager ---
    try:
        db_manager = DBManager()
        logger.info("DBManager initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize DBManager: {e}", exc_info=True)
        sys.exit("Application startup failed due to database error.")

    # --- Initialize PDFParser (now requires LLMInterface) ---
    pdf_parser = PDFParser(llm_interface)
    logger.info("PDFParser initialized successfully.")

    # --- Initialize Retriever and Generator ---
    retriever = Retriever(db_manager, embedding_model)
    generator = Generator(llm_interface)
    logger.info("RAG components (Retriever, Generator) initialized successfully.")

    # --- Prepare a dummy PDF for testing ---
    test_pdf_name = "test_paper.pdf"
    test_pdf_path = os.path.join(papers_dir, test_pdf_name)
    if not os.path.exists(test_pdf_path):
        logger.error(f"Test PDF not found at {test_pdf_path}. Please place a dummy PDF there to proceed.")
        sys.exit("Missing test PDF. Cannot proceed with tests.")

    # --- Test Setup: Clean DB and add a paper with sections and embeddings ---
    logger.info("\n--- Setting up test data: Cleaning DB and adding a paper ---")
    if os.path.exists(db_manager.db_path):
        os.remove(db_manager.db_path)
        logger.info(f"Removed existing database file: {db_manager.db_path}")
        db_manager._initialize_db()  # Re-initialize after deletion

    llm_extracted_metadata = pdf_parser.extract_metadata_with_llm(test_pdf_path)
    if not llm_extracted_metadata:
        logger.error("Failed to extract metadata using LLM. Cannot add paper to DB. Exiting test.")
        sys.exit("LLM metadata extraction failed.")

    paper_title = llm_extracted_metadata.get("title", "Untitled Paper")
    paper_authors = llm_extracted_metadata.get("authors", "Unknown Authors")
    paper_abstract = llm_extracted_metadata.get("abstract", "No abstract extracted.")
    paper_abstract_summary = llm_extracted_metadata.get("abstract_summary", "No abstract summary.")
    paper_year = 2023  # Placeholder for now

    paper_id = db_manager.add_paper(
        title=paper_title,
        authors=paper_authors,
        publication_year=paper_year,
        abstract=paper_abstract,
        file_path=test_pdf_path,
    )
    if not paper_id:
        logger.error("Failed to add Test Paper. Exiting.")
        sys.exit("Failed to add paper to DB, cannot proceed.")
    logger.info(
        f"Test Paper added with ID: {paper_id} using LLM-extracted metadata. Abstract Summary: {paper_abstract_summary}"
    )

    sections_data = pdf_parser.extract_sections_from_pdf(test_pdf_path, chunk_size=500, overlap=100)
    if sections_data:
        logger.info(f"Extracted {len(sections_data)} sections from PDF.")
        section_contents = [sec["content"] for sec in sections_data]
        embeddings = embedding_model.get_embedding(section_contents)

        if embeddings is not None and embeddings.shape[0] == len(sections_data):
            for i, section in enumerate(sections_data):
                db_manager.add_section(
                    paper_id=paper_id,
                    section_title=section["section_title"],
                    content=section["content"],
                    page_number=section["page_number"],
                    embedding=embeddings[i],
                )
            logger.info(f"Stored {len(sections_data)} sections with embeddings for paper ID {paper_id}.")
        else:
            logger.error("Failed to generate or match embeddings for all sections. Sections not stored with embeddings.")
    else:
        logger.warning("No sections extracted from PDF. RAG will not function.")

    # --- RAG Test ---
    logger.info("\n--- Performing RAG Query Test ---")
    sample_query = "What is the main idea of this paper?"  # Adjust based on your test PDF content
    logger.info(f"User Query: '{sample_query}'")

    retrieved_sections = retriever.retrieve_relevant_sections(sample_query, top_k=3)

    if retrieved_sections:
        logger.info(f"Retrieved {len(retrieved_sections)} sections. Generating answer...")
        rag_answer = generator.generate_answer(sample_query, retrieved_sections)
        print("\n--- RAG Answer ---")
        print(rag_answer)
        print("------------------\n")
    else:
        logger.warning("No sections retrieved, cannot generate RAG answer.")
        print("\n--- RAG Answer ---")
        print("I couldn't find any relevant information in your papers to answer that question.")
        print("------------------\n")

    logger.info("\n--- RAG test completed. ---")
    logger.info("Paper Agent started successfully. (CLI not yet implemented)")


if __name__ == "__main__":
    main()
