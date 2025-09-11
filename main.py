# paper_agent/main.py

import os
from utils.config import config
from utils.logger import logger
from database.db_manager import DBManager
from parsers.pdf_parser import PDFParser
from llm.llm_interface import LLMInterface
from embeddings.embedding_model import EmbeddingModel  # Import EmbeddingModel
import numpy as np  # Still need numpy for handling embeddings
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

    # --- Prepare a dummy PDF for testing ---
    test_pdf_name = "test_paper.pdf"
    test_pdf_path = os.path.join(papers_dir, test_pdf_name)
    if not os.path.exists(test_pdf_path):
        logger.error(f"Test PDF not found at {test_pdf_path}. Please place a dummy PDF there to proceed.")
        sys.exit("Missing test PDF. Cannot proceed with parser testing.")

    # --- Basic DBManager and LLM-Enhanced PDFParser Test ---
    logger.info("\n--- Testing DBManager, LLM-Enhanced PDFParser, and EmbeddingModel operations ---")

    # Clean up previous test data if any, for a fresh run
    if os.path.exists(db_manager.db_path):
        os.remove(db_manager.db_path)
        logger.info(f"Removed existing database file: {db_manager.db_path}")
        db_manager._initialize_db()  # Re-initialize after deletion

    # 1. Extract metadata using LLM
    logger.info(f"\n--- Extracting metadata from {test_pdf_path} using LLM ---")
    llm_extracted_metadata = pdf_parser.extract_metadata_with_llm(test_pdf_path)

    if not llm_extracted_metadata:
        logger.error("Failed to extract metadata using LLM. Cannot add paper to DB. Exiting test.")
        sys.exit("LLM metadata extraction failed.")

    # Use extracted metadata to add paper
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
    if paper_id:
        logger.info(f"Test Paper added with ID: {paper_id} using LLM-extracted metadata.")
        logger.info(f"LLM Abstract Summary: {paper_abstract_summary}")
    else:
        logger.error("Failed to add Test Paper.")
        sys.exit("Failed to add paper to DB, cannot proceed.")

    # 2. Extract sections and store them in the database with real embeddings
    logger.info("\n--- Extracting and Storing Sections with Embeddings ---")
    sections_data = pdf_parser.extract_sections_from_pdf(test_pdf_path, chunk_size=500, overlap=100)
    if sections_data:
        logger.info(f"Extracted {len(sections_data)} sections from PDF.")

        # Get all content for batch embedding
        section_contents = [sec["content"] for sec in sections_data]

        # Generate embeddings in a batch for efficiency
        logger.info(f"Generating {len(section_contents)} embeddings in batch...")
        embeddings = embedding_model.get_embedding(section_contents)

        if embeddings is not None and embeddings.shape[0] == len(sections_data):
            logger.info(f"Embeddings generated with shape: {embeddings.shape}")
            for i, section in enumerate(sections_data):
                section_id = db_manager.add_section(
                    paper_id=paper_id,
                    section_title=section["section_title"],
                    content=section["content"],
                    page_number=section["page_number"],
                    embedding=embeddings[i],  # Use the actual embedding
                )
                if section_id:
                    logger.debug(f"Stored section {i+1} (ID: {section_id}, Page: {section['page_number']}) with embedding.")
                else:
                    logger.error(f"Failed to store section {i+1} for paper ID {paper_id}.")
        else:
            logger.error("Failed to generate or match embeddings for all sections. Storing sections without embeddings.")
            # Fallback: store sections without embeddings if embedding generation fails
            for i, section in enumerate(sections_data):
                section_id = db_manager.add_section(
                    paper_id=paper_id,
                    section_title=section["section_title"],
                    content=section["content"],
                    page_number=section["page_number"],
                    embedding=None,
                )
    else:
        logger.warning("No sections extracted from PDF.")

    # 3. Verify stored sections and their embeddings
    stored_sections = db_manager.get_sections_for_paper(paper_id)
    if stored_sections:
        logger.info(f"\nRetrieved {len(stored_sections)} sections from DB for paper ID {paper_id}.")
        for sec in stored_sections[:3]:  # Log first 3
            logger.info(
                f"  - Section ID: {sec['id']}, Title: {sec['section_title']}, Page: {sec['page_number']}, Content (first 100): {sec['content'][:100]}..."
            )
            if sec["embedding"] is not None:
                logger.info(f"    Embedding shape: {sec['embedding'].shape}, Type: {type(sec['embedding'])}")
                # Verify it's a numpy array, not just a BLOB
                assert isinstance(sec["embedding"], np.ndarray), "Embedding not converted back to numpy array!"
            else:
                logger.warning("    Embedding is None.")
    else:
        logger.warning(f"No sections retrieved from DB for paper ID {paper_id}.")

    # 4. Add tags and references (re-using old test code for completeness)
    tag_ml_id = db_manager.add_tag("Machine Learning")
    if paper_id and tag_ml_id:
        db_manager.add_paper_tag(paper_id, tag_ml_id)
    logger.info(f"Tags for paper {paper_id}: {[t['name'] for t in db_manager.get_tags_for_paper(paper_id)]}")

    if paper_id:
        db_manager.add_paper_reference(paper_id, "Foundational Algorithms", "Classic Author", 2000, "10.0000/classic.2000")
        logger.info(
            f"References for paper {paper_id}: {[r['cited_title'] for r in db_manager.get_paper_references_for_paper(paper_id)]}"
        )

    logger.info("\n--- All core component tests completed. ---")

    logger.info("Paper Agent started successfully. (CLI not yet implemented)")


if __name__ == "__main__":
    main()
