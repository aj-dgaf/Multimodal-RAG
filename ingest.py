"""
ingest.py

Parses PDFs into text chunks, tables (as markdown), and images.
Embeds text/tables with a sentence-transformer and images with CLIP.
Builds and saves FAISS indexes.

Usage:
    python ingest.py
"""
