# paper_agent/ui/cli.py

import os
from typing import List, Dict, Any, Optional

from management.paper_manager import PaperManager
from utils.logger import logger
from utils.config import config


class CLI:
    """
    Command Line Interface for the Paper Agent.
    Allows users to interact with the PaperManager.
    """

    def __init__(self, paper_manager: PaperManager):
        self.paper_manager = paper_manager
        self.commands = {
            "add": self.add_paper_command,
            "list": self.list_papers_command,
            "details": self.show_paper_details_command,
            "query": self.query_rag_command,
            "summarize": self.summarize_paper_command,
            "tag": self.tag_paper_command,
            "untag": self.untag_paper_command,
            "delete": self.delete_paper_command,
            "help": self.show_help,
            "exit": self.exit_cli,
        }
        logger.info("CLI initialized. Type 'help' for commands.")

    def run(self):
        """Starts the interactive CLI loop."""
        print("\n--- Welcome to Paper Agent CLI ---")
        print("Type 'help' for a list of commands.")
        print("Type 'exit' to quit.")
        while True:
            try:
                command_line = input("\nPaperAgent> ").strip()
                if not command_line:
                    continue

                parts = command_line.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in self.commands:
                    self.commands[command](args)
                else:
                    print(f"Unknown command: '{command}'. Type 'help' for a list of commands.")
            except EOFError:  # Handle Ctrl+D
                self.exit_cli("")
            except Exception as e:
                logger.error(f"An unexpected error occurred in CLI: {e}", exc_info=True)
                print(f"An error occurred: {e}")

    def show_help(self, args: str):
        """Displays available commands and their usage."""
        print("\n--- Available Commands ---")
        print("  add <path_to_pdf>                  : Add a new paper from a PDF file.")
        print("  list                               : List all papers in your library.")
        print("  details <paper_id>                 : Show detailed information for a specific paper.")
        print("  query <your_question>              : Ask a question about your papers (RAG).")
        print("  summarize <paper_id>               : Generate a comprehensive summary for a paper using LLM.")
        print("  tag <paper_id> <tag_name>          : Add a tag to a paper.")
        print("  untag <paper_id> <tag_name>        : Remove a tag from a paper.")
        print("  delete <paper_id>                  : Delete a paper and all its data.")
        print("  help                               : Show this help message.")
        print("  exit                               : Exit the CLI.")
        print("--------------------------")

    def add_paper_command(self, args: str):
        """Handles the 'add' command."""
        file_path = args.strip()
        if not file_path:
            print("Usage: add <path_to_pdf_file>")
            return

        # Resolve absolute path for consistency
        absolute_path = os.path.abspath(file_path)

        # Check if the file exists before attempting to add
        if not os.path.exists(absolute_path):
            print(f"Error: File not found at '{absolute_path}'.")
            return
        if not absolute_path.lower().endswith(".pdf"):
            print(f"Error: File '{absolute_path}' is not a PDF.")
            return

        print(f"Attempting to add '{absolute_path}'...")
        paper_id = self.paper_manager.add_paper_from_file(absolute_path)
        if paper_id:
            print(f"Paper successfully added with ID: {paper_id}")
        else:
            print("Failed to add paper. Check logs for details.")

    def list_papers_command(self, args: str):
        """Handles the 'list' command."""
        papers = self.paper_manager.list_all_papers()
        if not papers:
            print("Your library is empty. Use 'add <path_to_pdf>' to add papers.")
            return

        print("\n--- Your Paper Library ---")
        for paper in papers:
            print(f"  ID: {paper['id']}")
            print(f"  Title: {paper['title']}")
            print(f"  Authors: {paper['authors']}")
            print(f"  Year: {paper['publication_year']}")
            print(f"  Path: {paper['file_path']}")
            print(f"  Summarized: {'Yes' if paper['is_summarized'] else 'No'}")
            print("-" * 30)
        print(f"Total papers: {len(papers)}")

    def show_paper_details_command(self, args: str):
        """Handles the 'details' command."""
        try:
            paper_id = int(args.strip())
        except ValueError:
            print("Usage: details <paper_id> (e.g., details 1)")
            return

        paper = self.paper_manager.get_paper_details(paper_id)
        if paper:
            print(f"\n--- Details for Paper ID: {paper['id']} ---")
            print(f"  Title: {paper['title']}")
            print(f"  Authors: {paper['authors']}")
            print(f"  Year: {paper['publication_year']}")
            print(f"  Abstract:\n{paper['abstract']}")
            print(f"  File Path: {paper['file_path']}")
            print(f"  Added Date: {paper['added_date']}")
            print(f"  DOI: {paper['doi'] if paper['doi'] else 'N/A'}")
            print(f"  URL: {paper['url'] if paper['url'] else 'N/A'}")
            print(f"  Summarized: {'Yes' if paper['is_summarized'] else 'No'}")
            if paper["is_summarized"]:
                print(f"  Summary:\n{paper['summary_text']}")
            print(f"  Tags: {', '.join([t['name'] for t in paper['tags']]) if paper['tags'] else 'None'}")
            print(f"  References ({len(paper['references'])}):")
            for ref in paper["references"]:
                print(
                    f"    - {ref['cited_title']} ({ref['cited_year']}) by {ref['cited_authors']} (In Library: {'Yes' if ref['is_in_library'] else 'No'})"
                )
            print("---------------------------------------")
        else:
            print(f"Paper with ID {paper_id} not found.")

    def query_rag_command(self, args: str):
        """Handles the 'query' command."""
        query_text = args.strip()
        if not query_text:
            print("Usage: query <your_question>")
            return

        print(f"Querying your papers for: '{query_text}'...")
        result = self.paper_manager.query_papers_rag(query_text)

        print("\n--- RAG Answer ---")
        print(result["answer"])
        print("\n--- Sources Used ---")
        if result["retrieved_sections"]:
            for i, section in enumerate(result["retrieved_sections"]):
                print(f"  {i+1}. Paper: {section['paper_title']} (ID: {section['paper_id']})")
                print(f"     Section: {section['section_title']} (Page: {section['page_number']})")
                print(f"     Score: {section['score']:.4f}")
                # print(f"     Content (snippet): {section['content'][:200]}...") # Optional: show snippet
        else:
            print("No relevant sections found to answer the query.")
        print("--------------------\n")

    def summarize_paper_command(self, args: str):
        """Handles the 'summarize' command."""
        try:
            paper_id = int(args.strip())
        except ValueError:
            print("Usage: summarize <paper_id> (e.g., summarize 1)")
            return

        paper = self.paper_manager.db_manager.get_paper(paper_id)
        if not paper:
            print(f"Paper with ID {paper_id} not found.")
            return

        if paper["is_summarized"]:
            print(f"Paper ID {paper_id} is already summarized.")
            # Optionally, ask if they want to re-summarize
            # confirm = input("Re-summarize? (y/N): ").strip().lower()
            # if confirm != 'y':
            #     print("Summarization skipped.")
            #     return

        print(f"Generating summary for paper ID {paper_id}. This may take a while...")
        summary = self.paper_manager.summarize_paper(paper_id)
        if summary:
            print("\n--- Generated Summary ---")
            print(summary)
            print("-------------------------\n")
        else:
            print("Failed to generate summary. Check logs for details.")

    def tag_paper_command(self, args: str):
        """Handles the 'tag' command."""
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: tag <paper_id> <tag_name>")
            return

        try:
            paper_id = int(parts[0])
            tag_name = parts[1].strip()
        except ValueError:
            print("Usage: tag <paper_id> <tag_name>")
            return

        if self.paper_manager.add_tag_to_paper(paper_id, tag_name):
            print(f"Tag '{tag_name}' added to paper ID {paper_id}.")
        else:
            print(f"Failed to add tag '{tag_name}' to paper ID {paper_id}.")

    def untag_paper_command(self, args: str):
        """Handles the 'untag' command."""
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: untag <paper_id> <tag_name>")
            return

        try:
            paper_id = int(parts[0])
            tag_name = parts[1].strip()
        except ValueError:
            print("Usage: untag <paper_id> <tag_name>")
            return

        if self.paper_manager.remove_tag_from_paper(paper_id, tag_name):
            print(f"Tag '{tag_name}' removed from paper ID {paper_id}.")
        else:
            print(f"Failed to remove tag '{tag_name}' from paper ID {paper_id}.")

    def delete_paper_command(self, args: str):
        """Handles the 'delete' command."""
        try:
            paper_id = int(args.strip())
        except ValueError:
            print("Usage: delete <paper_id> (e.g., delete 1)")
            return

        confirm = (
            input(f"Are you sure you want to delete paper ID {paper_id} and all its associated data? (y/N): ")
            .strip()
            .lower()
        )
        if confirm == "y":
            if self.paper_manager.delete_paper(paper_id):
                print(f"Paper ID {paper_id} successfully deleted.")
            else:
                print(f"Failed to delete paper ID {paper_id}.")
        else:
            print("Deletion cancelled.")

    def exit_cli(self, args: str):
        """Exits the CLI."""
        print("Exiting Paper Agent. Goodbye!")
        exit()
