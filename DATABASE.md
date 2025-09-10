## Database Schema Design

### papers: Stores core metadata for each paper.

id (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for the paper.
title (TEXT NOT NULL): Title of the paper.
authors (TEXT): Comma-separated list of authors.
publication_year (INTEGER): Year of publication.
abstract (TEXT): Abstract of the paper.
file_path (TEXT NOT NULL UNIQUE): Absolute path to the PDF file on disk.
added_date (TEXT NOT NULL): Date when the paper was added to the system (ISO format).
is_summarized (INTEGER DEFAULT 0): Boolean (0 or 1) indicating if a summary has been generated.
summary_text (TEXT): The generated summary text.
doi (TEXT UNIQUE): Digital Object Identifier (if available).
url (TEXT): URL to the paper (e.g., ArXiv, publisher).

### tags: Stores unique tags.

id (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for the tag.
name (TEXT NOT NULL UNIQUE): Name of the tag (e.g., "NLP", "Computer Vision", "RAG").

### paper_tags: A many-to-many relationship table between papers and tags.

paper_id (INTEGER NOT NULL): Foreign key to papers.id.
tag_id (INTEGER NOT NULL): Foreign key to tags.id.
PRIMARY KEY (paper_id, tag_id): Composite primary key to ensure unique relationships.

### sections: Stores parsed sections of the paper for RAG.

id (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for the section.
paper_id (INTEGER NOT NULL): Foreign key to papers.id.
section_title (TEXT): Title of the section (e.g., "Introduction", "Methodology").
content (TEXT NOT NULL): The textual content of the section.
page_number (INTEGER): The page number where the section starts.
embedding (BLOB): Binary representation of the embedding vector for this section. (We'll use BLOB for now, and convert from/to numpy arrays).

### references: Stores references cited within a paper.

id (INTEGER PRIMARY KEY AUTOINCREMENT): Unique identifier for the reference.
citing_paper_id (INTEGER NOT NULL): Foreign key to papers.id (the paper that cites this reference).
cited_title (TEXT): Title of the cited work.
cited_authors (TEXT): Authors of the cited work.
cited_year (INTEGER): Publication year of the cited work.
cited_doi (TEXT): DOI of the cited work (if available).
cited_url (TEXT): URL of the cited work.
is_in_library (INTEGER DEFAULT 0): Boolean (0 or 1) indicating if this cited paper is also in our papers table.