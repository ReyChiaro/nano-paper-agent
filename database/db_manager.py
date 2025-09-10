# paper_agent/database/db_manager.py

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np # For handling embeddings

from utils.config import config
from utils.logger import logger

class DBManager:
    """
    Manages all database interactions for the Paper Agent.
    Uses SQLite and handles connection, schema creation, and CRUD operations.
    """
    def __init__(self):
        db_dir = config.get("DB_DIR")
        db_name = config.get("DATABASE_NAME")
        self.db_path = os.path.join(db_dir, db_name)
        self._initialize_db()
        logger.info(f"Database initialized at: {self.db_path}")

    def _initialize_db(self):
        """
        Ensures the database directory exists and creates tables if they don't.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            self.close_connection(conn)

    def get_connection(self) -> sqlite3.Connection:
        """
        Establishes and returns a database connection.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row # Allows accessing columns by name
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def close_connection(self, conn: sqlite3.Connection):
        """
        Closes a database connection.
        """
        if conn:
            conn.close()

    def _execute_query(self, query: str, params: tuple = ()) -> Optional[List[sqlite3.Row]]:
        """
        Helper method to execute a read query and return results.
        Returns None on error.
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database query failed: {query} with params {params}. Error: {e}")
            return None
        finally:
            self.close_connection(conn)

    def _execute_update(self, query: str, params: tuple = ()) -> Optional[int]:
        """
        Helper method to execute an insert/update/delete query.
        Returns the last row ID for inserts or number of affected rows for updates/deletes.
        Returns None on error.
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            if query.strip().upper().startswith("INSERT"):
                return cursor.lastrowid
            return cursor.rowcount
        except sqlite3.IntegrityError as e:
            logger.warning(f"Database integrity error: {e}. Query: {query} with params {params}")
            return None # Indicate failure due to integrity constraint
        except sqlite3.Error as e:
            logger.error(f"Database update failed: {query} with params {params}. Error: {e}")
            return None
        finally:
            self.close_connection(conn)

    # --- Paper Operations ---

    def add_paper(self,
                  title: str,
                  file_path: str,
                  authors: Optional[str] = None,
                  publication_year: Optional[int] = None,
                  abstract: Optional[str] = None,
                  doi: Optional[str] = None,
                  url: Optional[str] = None) -> Optional[int]:
        """
        Adds a new paper to the database.
        Returns the ID of the newly added paper, or None if failed.
        """
        added_date = datetime.now().isoformat(sep=' ', timespec='seconds')
        query = """
        INSERT INTO papers (title, authors, publication_year, abstract, file_path, added_date, doi, url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (title, authors, publication_year, abstract, file_path, added_date, doi, url)
        paper_id = self._execute_update(query, params)
        if paper_id:
            logger.info(f"Added paper: '{title}' (ID: {paper_id})")
        return paper_id

    def get_paper(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a paper by its ID.
        Returns a dictionary representing the paper, or None if not found.
        """
        query = "SELECT * FROM papers WHERE id = ?"
        result = self._execute_query(query, (paper_id,))
        return dict(result[0]) if result else None

    def get_paper_by_filepath(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a paper by its file path.
        Returns a dictionary representing the paper, or None if not found.
        """
        query = "SELECT * FROM papers WHERE file_path = ?"
        result = self._execute_query(query, (file_path,))
        return dict(result[0]) if result else None

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Retrieves all papers from the database.
        """
        query = "SELECT * FROM papers ORDER BY added_date DESC"
        results = self._execute_query(query)
        return [dict(row) for row in results] if results else []

    def update_paper_summary(self, paper_id: int, summary_text: str) -> bool:
        """
        Updates the summary text and sets is_summarized to true for a paper.
        Returns True on success, False otherwise.
        """
        query = "UPDATE papers SET summary_text = ?, is_summarized = 1 WHERE id = ?"
        rows_affected = self._execute_update(query, (summary_text, paper_id))
        if rows_affected == 1:
            logger.info(f"Updated summary for paper ID: {paper_id}")
            return True
        logger.warning(f"Failed to update summary for paper ID: {paper_id}. Rows affected: {rows_affected}")
        return False

    def delete_paper(self, paper_id: int) -> bool:
        """
        Deletes a paper and all its associated data (tags, sections, references)
        due to CASCADE DELETE constraints.
        Returns True on success, False otherwise.
        """
        query = "DELETE FROM papers WHERE id = ?"
        rows_affected = self._execute_update(query, (paper_id,))
        if rows_affected == 1:
            logger.info(f"Deleted paper with ID: {paper_id}")
            return True
        logger.warning(f"Failed to delete paper with ID: {paper_id}. Rows affected: {rows_affected}")
        return False

    # --- Tag Operations ---

    def add_tag(self, name: str) -> Optional[int]:
        """
        Adds a new tag if it doesn't exist.
        Returns the ID of the tag, or None if failed.
        """
        # Check if tag already exists
        existing_tag = self.get_tag_by_name(name)
        if existing_tag:
            return existing_tag['id']

        query = "INSERT INTO tags (name) VALUES (?)"
        tag_id = self._execute_update(query, (name,))
        if tag_id:
            logger.info(f"Added tag: '{name}' (ID: {tag_id})")
        return tag_id

    def get_tag_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a tag by its name.
        """
        query = "SELECT * FROM tags WHERE name = ?"
        result = self._execute_query(query, (name,))
        return dict(result[0]) if result else None

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """
        Retrieves all tags.
        """
        query = "SELECT * FROM tags ORDER BY name ASC"
        results = self._execute_query(query)
        return [dict(row) for row in results] if results else []

    def add_paper_tag(self, paper_id: int, tag_id: int) -> bool:
        """
        Associates a tag with a paper.
        Returns True on success, False if already exists or failed.
        """
        query = "INSERT OR IGNORE INTO paper_tags (paper_id, tag_id) VALUES (?, ?)"
        rows_affected = self._execute_update(query, (paper_id, tag_id))
        if rows_affected is not None: # It can be 0 if already exists, which is not an error
            if rows_affected == 1:
                logger.info(f"Associated paper {paper_id} with tag {tag_id}")
            return True
        return False

    def get_tags_for_paper(self, paper_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves all tags associated with a specific paper.
        """
        query = """
        SELECT t.id, t.name
        FROM tags t
        JOIN paper_tags pt ON t.id = pt.tag_id
        WHERE pt.paper_id = ?
        ORDER BY t.name ASC
        """
        results = self._execute_query(query, (paper_id,))
        return [dict(row) for row in results] if results else []

    def remove_paper_tag(self, paper_id: int, tag_id: int) -> bool:
        """
        Removes a tag association from a paper.
        Returns True on success, False otherwise.
        """
        query = "DELETE FROM paper_tags WHERE paper_id = ? AND tag_id = ?"
        rows_affected = self._execute_update(query, (paper_id, tag_id))
        if rows_affected == 1:
            logger.info(f"Removed tag {tag_id} from paper {paper_id}")
            return True
        logger.warning(f"Failed to remove tag {tag_id} from paper {paper_id}. Rows affected: {rows_affected}")
        return False

    # --- Section Operations ---

    def add_section(self, paper_id: int, section_title: str, content: str,
                    page_number: Optional[int] = None, embedding: Optional[np.ndarray] = None) -> Optional[int]:
        """
        Adds a parsed section of a paper.
        Embedding is stored as BLOB.
        """
        embedding_blob = embedding.tobytes() if embedding is not None else None
        query = "INSERT INTO sections (paper_id, section_title, content, page_number, embedding) VALUES (?, ?, ?, ?, ?)"
        params = (paper_id, section_title, content, page_number, embedding_blob)
        section_id = self._execute_update(query, params)
        if section_id:
            logger.debug(f"Added section for paper {paper_id}: '{section_title}' (ID: {section_id})")
        return section_id

    def get_sections_for_paper(self, paper_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves all sections for a given paper.
        Converts embedding BLOB back to numpy array.
        """
        query = "SELECT id, paper_id, section_title, content, page_number, embedding FROM sections WHERE paper_id = ? ORDER BY page_number ASC"
        results = self._execute_query(query, (paper_id,))
        sections = []
        if results:
            for row in results:
                section_data = dict(row)
                if section_data['embedding']:
                    section_data['embedding'] = np.frombuffer(section_data['embedding'], dtype=np.float32) # Assuming float32
                sections.append(section_data)
        return sections

    def get_section_by_id(self, section_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single section by its ID.
        Converts embedding BLOB back to numpy array.
        """
        query = "SELECT id, paper_id, section_title, content, page_number, embedding FROM sections WHERE id = ?"
        result = self._execute_query(query, (section_id,))
        if result:
            section_data = dict(result[0])
            if section_data['embedding']:
                section_data['embedding'] = np.frombuffer(section_data['embedding'], dtype=np.float32)
            return section_data
        return None

    def delete_sections_for_paper(self, paper_id: int) -> bool:
        """
        Deletes all sections associated with a specific paper.
        Returns True on success, False otherwise.
        """
        query = "DELETE FROM sections WHERE paper_id = ?"
        rows_affected = self._execute_update(query, (paper_id,))
        if rows_affected is not None:
            logger.info(f"Deleted {rows_affected} sections for paper ID: {paper_id}")
            return True
        return False

    # --- Reference Operations ---

    def add_reference(self,
                      citing_paper_id: int,
                      cited_title: Optional[str] = None,
                      cited_authors: Optional[str] = None,
                      cited_year: Optional[int] = None,
                      cited_doi: Optional[str] = None,
                      cited_url: Optional[str] = None,
                      is_in_library: bool = False
                      ) -> Optional[int]:
        """
        Adds a reference cited by a paper.
        Returns the ID of the newly added reference, or None if failed.
        """
        query = """
        INSERT INTO references (citing_paper_id, cited_title, cited_authors, cited_year, cited_doi, cited_url, is_in_library)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (citing_paper_id, cited_title, cited_authors, cited_year, cited_doi, cited_url, 1 if is_in_library else 0)
        ref_id = self._execute_update(query, params)
        if ref_id:
            logger.debug(f"Added reference for paper {citing_paper_id}: '{cited_title}' (ID: {ref_id})")
        return ref_id

    def get_references_for_paper(self, citing_paper_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves all references cited by a specific paper.
        """
        query = "SELECT * FROM references WHERE citing_paper_id = ? ORDER BY cited_title ASC"
        results = self._execute_query(query, (citing_paper_id,))
        return [dict(row) for row in results] if results else []

    def update_reference_in_library_status(self, ref_id: int, is_in_library: bool) -> bool:
        """
        Updates the 'is_in_library' status for a specific reference.
        Returns True on success, False otherwise.
        """
        query = "UPDATE references SET is_in_library = ? WHERE id = ?"
        rows_affected = self._execute_update(query, (1 if is_in_library else 0, ref_id))
        if rows_affected == 1:
            logger.info(f"Updated is_in_library status for reference ID: {ref_id} to {is_in_library}")
            return True
        logger.warning(f"Failed to update is_in_library status for reference ID: {ref_id}. Rows affected: {rows_affected}")
        return False

    def get_papers_by_tag(self, tag_name: str) -> List[Dict[str, Any]]:
        """
        Retrieves all papers associated with a specific tag name.
        """
        query = """
        SELECT p.*
        FROM papers p
        JOIN paper_tags pt ON p.id = pt.paper_id
        JOIN tags t ON pt.tag_id = t.id
        WHERE t.name = ?
        ORDER BY p.title ASC
        """
        results = self._execute_query(query, (tag_name,))
        return [dict(row) for row in results] if results else []

