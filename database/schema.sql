-- paper_agent/database/schema.sql

-- Table to store core paper metadata
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    authors TEXT, -- Comma-separated authors
    publication_year INTEGER,
    abstract TEXT,
    file_path TEXT NOT NULL UNIQUE, -- Absolute path to the PDF
    added_date TEXT NOT NULL, -- ISO format date (YYYY-MM-DD HH:MM:SS)
    is_summarized INTEGER DEFAULT 0, -- 0 for false, 1 for true
    summary_text TEXT,
    doi TEXT UNIQUE,
    url TEXT
);

-- Table to store unique tags
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

-- Junction table for many-to-many relationship between papers and tags
CREATE TABLE IF NOT EXISTS paper_tags (
    paper_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (paper_id, tag_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Table to store parsed sections of papers for RAG
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    section_title TEXT, -- e.g., "Introduction", "Methodology"
    content TEXT NOT NULL,
    page_number INTEGER,
    embedding BLOB, -- Store numpy array as BLOB
    FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
);

-- Table to store references cited within a paper
CREATE TABLE IF NOT EXISTS references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    citing_paper_id INTEGER NOT NULL, -- The paper that contains this reference
    cited_title TEXT,
    cited_authors TEXT,
    cited_year INTEGER,
    cited_doi TEXT,
    cited_url TEXT,
    is_in_library INTEGER DEFAULT 0, -- 0 if not in our papers table, 1 if it is
    FOREIGN KEY (citing_paper_id) REFERENCES papers(id) ON DELETE CASCADE
);
