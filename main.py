# paper_agent/main.py

import os
from utils.config import config
from utils.logger import logger
from database.db_manager import DBManager
from parsers.pdf_parser import PDFParser
from llm.llm_interface import LLMInterface
from embeddings.embedding_model import EmbeddingModel
from rag.retriever import Retriever
from rag.generator import Generator
from management.paper_manager import PaperManager  # Import PaperManager
import numpy as np
import sys


def main():
    """
    Main entry point for the Paper Agent application.
    Initializes core components and uses PaperManager for end-to-end testing.
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
    llm_interface = LLMInterface()
    embedding_model = None
    try:
        embedding_model = EmbeddingModel()
    except Exception as e:
        logger.critical(f"Failed to initialize EmbeddingModel: {e}", exc_info=True)
        sys.exit("Application startup failed due to embedding model error.")

    db_manager = None
    try:
        db_manager = DBManager()
    except Exception as e:
        logger.critical(f"Failed to initialize DBManager: {e}", exc_info=True)
        sys.exit("Application startup failed due to database error.")

    pdf_parser = PDFParser(llm_interface)
    retriever = Retriever(db_manager, embedding_model)
    generator = Generator(llm_interface)

    # --- Initialize PaperManager ---
    paper_manager = PaperManager(db_manager, pdf_parser, embedding_model, retriever, generator, llm_interface)
    logger.info("PaperManager initialized successfully.")

    # --- Prepare a dummy PDF for testing ---
    test_pdf_name = "test_paper.pdf"
    test_pdf_path = os.path.join(papers_dir, test_pdf_name)
    if not os.path.exists(test_pdf_path):
        logger.error(f"Test PDF not found at {test_pdf_path}. Please place a dummy PDF there to proceed.")
        sys.exit("Missing test PDF. Cannot proceed with tests.")

    # --- End-to-End Test with PaperManager ---
    logger.info("\n--- Starting End-to-End Test with PaperManager ---")

    # Clean up previous test data for a fresh run
    if os.path.exists(db_manager.db_path):
        os.remove(db_manager.db_path)
        logger.info(f"Removed existing database file: {db_manager.db_path}")
        db_manager._initialize_db()  # Re-initialize after deletion

    # 1. Add a paper
    logger.info("\n--- Test 1: Adding a new paper ---")
    paper_id = paper_manager.add_paper_from_file(test_pdf_path)
    if paper_id:
        logger.info(f"Successfully added paper from '{test_pdf_path}' with ID: {paper_id}")
    else:
        logger.error(f"Failed to add paper from '{test_pdf_path}'.")
        sys.exit("Failed to add paper, cannot proceed with further tests.")

    # 2. List all papers
    logger.info("\n--- Test 2: Listing all papers ---")
    all_papers = paper_manager.list_all_papers()
    if all_papers:
        logger.info(f"Found {len(all_papers)} paper(s) in the library:")
        for paper in all_papers:
            logger.info(f"  ID: {paper['id']}, Title: {paper['title']}, Authors: {paper['authors']}")
    else:
        logger.info("No papers found in the library.")

    # 3. Get paper details
    logger.info(f"\n--- Test 3: Getting details for paper ID {paper_id} ---")
    paper_details = paper_manager.get_paper_details(paper_id)
    if paper_details:
        logger.info(f"Details for Paper ID {paper_id}:")
        logger.info(f"  Title: {paper_details['title']}")
        logger.info(f"  Abstract: {paper_details['abstract'][:100]}...")
        logger.info(f"  Tags: {[t['name'] for t in paper_details['tags']]}")
        logger.info(f"  References: {[r['cited_title'] for r in paper_details['references']]}")
    else:
        logger.warning(f"Could not retrieve details for paper ID {paper_id}.")

    # 4. Add a tag to the paper
    logger.info(f"\n--- Test 4: Adding a tag to paper ID {paper_id} ---")
    tag_name = "TestTag"
    if paper_manager.add_tag_to_paper(paper_id, tag_name):
        logger.info(f"Tag '{tag_name}' added to paper ID {paper_id}.")
        updated_paper_details = paper_manager.get_paper_details(paper_id)
        logger.info(f"  Updated Tags: {[t['name'] for t in updated_paper_details['tags']]}")
    else:
        logger.error(f"Failed to add tag '{tag_name}' to paper ID {paper_id}.")

    # 5. Summarize the paper
    logger.info(f"\n--- Test 5: Summarizing paper ID {paper_id} ---")
    summary = paper_manager.summarize_paper(paper_id)
    if summary:
        print("\n--- Generated Summary ---")
        print(summary)
        print("-------------------------\n")
        # Verify summary is in DB
        retrieved_paper = db_manager.get_paper(paper_id)
        if retrieved_paper and retrieved_paper["is_summarized"] and retrieved_paper["summary_text"]:
            logger.info("Summary successfully stored in DB.")
        else:
            logger.error("Summary not found/stored in DB after generation.")
    else:
        logger.warning(f"Failed to generate summary for paper ID {paper_id}.")

    # 6. Perform a RAG query
    logger.info("\n--- Test 6: Performing a RAG query ---")
    sample_query = "What is the primary contribution of this paper?"  # Tailor to your test PDF
    rag_result = paper_manager.query_papers_rag(sample_query, top_k_sections=3)
    print("\n--- RAG Query Result ---")
    print(f"Query: {rag_result['query']}")
    print(f"Answer: {rag_result['answer']}")
    print("\nRetrieved Sections (top 2 for brevity):")
    for i, section in enumerate(rag_result["retrieved_sections"][:2]):
        print(
            f"  {i+1}. Paper: {section['paper_title']}, Section: {section['section_title']} (Score: {section['score']:.4f})"
        )
        print(f"     Content: {section['content'][:150]}...")
    print("-------------------------\n")

    # 7. Delete the paper
    logger.info(f"\n--- Test 7: Deleting paper ID {paper_id} ---")
    if paper_manager.delete_paper(paper_id):
        logger.info(f"Successfully deleted paper ID {paper_id}.")
        # Verify deletion
        if not paper_manager.get_paper_details(paper_id):
            logger.info("Paper confirmed as deleted from DB.")
        else:
            logger.error("Paper still exists after deletion attempt.")
    else:
        logger.error(f"Failed to delete paper ID {paper_id}.")

    logger.info("\n--- End-to-End Test Completed. ---")
    logger.info("Paper Agent application has finished its current run.")


if __name__ == "__main__":
    main()
