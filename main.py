# paper_agent/main.py

import os
from utils.config import config
from utils.logger import logger
from database.db_manager import DBManager
from parsers.pdf_parser import PDFParser  # Import the new PDFParser
import numpy as np  # For testing embeddings
import sys  # For exiting if no test PDF


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

    # --- Initialize DBManager ---
    try:
        db_manager = DBManager()
        logger.info("DBManager initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize DBManager: {e}", exc_info=True)
        sys.exit("Application startup failed due to database error.")

    # --- Initialize PDFParser ---
    pdf_parser = PDFParser()
    logger.info("PDFParser initialized successfully.")

    # --- Prepare a dummy PDF for testing ---
    test_pdf_name = "test_paper.pdf"
    test_pdf_path = os.path.join(papers_dir, test_pdf_name)
    if not os.path.exists(test_pdf_path):
        logger.error(f"Test PDF not found at {test_pdf_path}. Please place a dummy PDF there to proceed.")
        sys.exit("Missing test PDF. Cannot proceed with parser testing.")

    # --- Basic DBManager and PDFParser Test ---
    logger.info("\n--- Testing DBManager and PDFParser operations ---")

    # Clean up previous test data if any, for a fresh run
    # (In a real app, you'd have proper data management, not just deleting the DB)
    if os.path.exists(db_manager.db_path):
        os.remove(db_manager.db_path)
        logger.info(f"Removed existing database file: {db_manager.db_path}")
        db_manager._initialize_db()  # Re-initialize after deletion

    # 1. Add a paper (using test PDF path)
    paper_title = "Sample Research Paper on AI"
    paper_authors = "A. Researcher, B. Developer"
    paper_year = 2023
    paper_abstract = "This is a sample abstract for a research paper, demonstrating the capabilities of the PDF parser."

    paper_id = db_manager.add_paper(
        title=paper_title,
        authors=paper_authors,
        publication_year=paper_year,
        abstract=paper_abstract,
        file_path=test_pdf_path,
        doi="10.1234/sample.2023.1",
    )
    if paper_id:
        logger.info(f"Test Paper added with ID: {paper_id}")
    else:
        logger.error("Failed to add Test Paper.")
        sys.exit("Failed to add paper to DB, cannot proceed.")

    # 2. Extract metadata and text from the test PDF
    logger.info(f"\n--- Parsing PDF: {test_pdf_path} ---")
    metadata = pdf_parser.extract_metadata_from_pdf(test_pdf_path)
    logger.info(f"Extracted Metadata: {metadata}")

    # Optional: Update paper title/authors from PDF metadata if more accurate
    # For now, we'll stick to what we manually provided in add_paper for simplicity
    # if metadata.get('title'):
    #     # A method to update paper metadata would be needed here
    #     pass

    full_text_content = pdf_parser.extract_text_from_pdf(test_pdf_path)
    if full_text_content:
        logger.info(f"Extracted {len(full_text_content)} characters of full text.")
        # logger.debug(f"Full text (first 500 chars): {full_text_content[:500]}...")
    else:
        logger.warning(f"Could not extract full text from {test_pdf_path}.")

    # 3. Extract sections and store them in the database
    logger.info("\n--- Extracting and Storing Sections ---")
    sections_data = pdf_parser.extract_sections_from_pdf(test_pdf_path, chunk_size=500, overlap=100)
    if sections_data:
        logger.info(f"Extracted {len(sections_data)} sections from PDF.")
        # For testing, let's add a dummy embedding
        dummy_embedding_dim = 768  # Standard for many sentence transformers
        for i, section in enumerate(sections_data):
            # Simulate embedding generation (will be real in next step)
            dummy_embedding = np.random.rand(dummy_embedding_dim).astype(np.float32)
            section_id = db_manager.add_section(
                paper_id=paper_id,
                section_title=section["section_title"],
                content=section["content"],
                page_number=section["page_number"],
                embedding=dummy_embedding,
            )
            if section_id:
                logger.debug(f"Stored section {i+1} (ID: {section_id}, Page: {section['page_number']})")
            else:
                logger.error(f"Failed to store section {i+1} for paper ID {paper_id}.")
    else:
        logger.warning("No sections extracted from PDF.")

    # 4. Verify stored sections
    stored_sections = db_manager.get_sections_for_paper(paper_id)
    if stored_sections:
        logger.info(f"\nRetrieved {len(stored_sections)} sections from DB for paper ID {paper_id}.")
        for sec in stored_sections[:3]:  # Log first 3
            logger.info(
                f"  - Section ID: {sec['id']}, Title: {sec['section_title']}, Page: {sec['page_number']}, Content (first 100): {sec['content'][:100]}..."
            )
            if sec["embedding"] is not None:
                logger.info(f"    Embedding shape: {sec['embedding'].shape}")
            else:
                logger.warning("    Embedding is None.")
    else:
        logger.warning(f"No sections retrieved from DB for paper ID {paper_id}.")

    # 5. Add tags and references (re-using old test code for completeness, but focus is parser)
    tag_ml_id = db_manager.add_tag("Machine Learning")
    if paper_id and tag_ml_id:
        db_manager.add_paper_tag(paper_id, tag_ml_id)
    logger.info(f"Tags for paper {paper_id}: {[t['name'] for t in db_manager.get_tags_for_paper(paper_id)]}")

    if paper_id:
        db_manager.add_paper_reference(paper_id, "Foundational Algorithms", "Classic Author", 2000, "10.0000/classic.2000")
        logger.info(
            f"References for paper {paper_id}: {[r['cited_title'] for r in db_manager.get_paper_references_for_paper(paper_id)]}"
        )

    logger.info("\n--- DBManager and PDFParser tests completed. ---")

    # --- Future: Initialize other modules ---
    # from embeddings.embedding_model import EmbeddingModel
    # from rag.retriever import Retriever
    # from llm.llm_interface import LLMInterface
    # from management.paper_manager import PaperManager
    # from ui.cli import CLI

    # db_manager = DBManager() # Already initialized
    # pdf_parser = PDFParser() # Already initialized
    # embedding_model = EmbeddingModel()
    # retriever = Retriever(db_manager, embedding_model)
    # llm_interface = LLMInterface()
    # paper_manager = PaperManager(db_manager, pdf_parser, embedding_model, retriever, llm_interface)
    # cli = CLI(paper_manager)

    # --- Start the CLI ---
    # cli.run()

    logger.info("Paper Agent started successfully. (CLI not yet implemented)")


if __name__ == "__main__":
    main()
