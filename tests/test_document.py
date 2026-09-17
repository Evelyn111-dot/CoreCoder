from pathlib import Path

from corecoder.rag.document import (
    Document,
    DocumentChunk,
    load_document,
    split_document,
)


ROOT_DIR = Path(__file__).parent.parent
TEST_DOCUMENT = ROOT_DIR / "knowledge" / "test.md"


def test_load_document():
    doc = load_document(str(TEST_DOCUMENT))

    assert isinstance(doc, Document)
    assert doc.content
    assert doc.source == str(TEST_DOCUMENT)
    assert doc.metadata["file_name"] == "test.md"


def test_split_document():
    doc = Document(
        content="abcdefghijklmnopqrstuvwxyz",
        source="test.txt",
    )

    chunks = split_document(
        doc,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.source == "test.txt"
        assert len(chunk.content) <= 10


def test_chunk_metadata():
    doc = Document(
        content="hello world",
        source="test.txt",
    )

    chunks = split_document(
        doc,
        chunk_size=5,
        chunk_overlap=1,
    )

    assert chunks
    assert chunks[0].metadata["chunk_index"] == 0


def test_markdown_chunker():
    doc = Document(
        content="""# Introduction

This is introduction.

## Installation

Install CoreCoder.

## Usage

Run the agent.
""",
        source="test.md",
    )

    chunks = split_document(
        doc,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert chunks
    assert chunks[0].metadata["section"] == "Introduction"


def test_code_chunker():
    doc = Document(
        content="""class Agent:

    def run(self):
        return "hello"

    def stop(self):
        return "stop"
""",
        source="agent.py",
    )

    chunks = split_document(
        doc,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert chunks

    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert "chunk_type" in chunk.metadata
        assert "symbol" in chunk.metadata


def test_invalid_chunk_size():
    doc = Document(
        content="hello",
        source="test.txt",
    )

    try:
        split_document(
            doc,
            chunk_size=0,
        )
        assert False
    except ValueError:
        assert True


def test_invalid_overlap():
    doc = Document(
        content="hello",
        source="test.txt",
    )

    try:
        split_document(
            doc,
            chunk_size=10,
            chunk_overlap=10,
        )
        assert False
    except ValueError:
        assert True