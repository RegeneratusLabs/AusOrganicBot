import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pdfplumber
from PyPDF2 import PdfReader

from app.data.html_converter import HTMLConverter


class DocumentLoader:
    """Load and process PDF documents for RAG with page tracking"""

    def __init__(self, data_dir: Path, generate_html: bool = True):
        self.data_dir = data_dir
        self.generate_html = generate_html
        self.html_converter = HTMLConverter(data_dir) if generate_html else None

    def load_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using pdfplumber (legacy method)"""
        text = ""

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"Failed to extract text from {pdf_path}: {e2}")
                raise

        return text.strip()

    def load_pdf_with_pages(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with page-by-page tracking"""
        pages = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    pages.append(
                        {
                            "page_number": page_num,
                            "text": page_text,
                            "char_count": len(page_text),
                        }
                    )
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path}: {e}")
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    pages.append(
                        {
                            "page_number": page_num,
                            "text": page_text,
                            "char_count": len(page_text),
                        }
                    )
            except Exception as e2:
                print(f"Failed to extract text from {pdf_path}: {e2}")
                raise

        return pages

    def chunk_text(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks (legacy method)"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            if end < text_length:
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:
                    chunk = text[start : start + break_point + 1]
                    end = start + break_point + 1

            candidate = chunk.strip()
            if candidate and len(candidate) >= 20:
                chunks.append(candidate)
            start = end - chunk_overlap

        return chunks

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Split pages into chunks with page tracking

        Returns list of chunks with:
        - text: chunk content
        - page_number: which page this chunk is from
        - char_start: start position within the page
        - char_end: end position within the page
        """
        chunks = []

        for page in pages:
            page_text = page["text"]
            page_num = page["page_number"]

            if not page_text.strip():
                continue

            page_chunks = self._chunk_single_page(page_text, chunk_size, chunk_overlap)

            for chunk_info in page_chunks:
                chunks.append(
                    {
                        "text": chunk_info["text"],
                        "page_number": page_num,
                        "char_start": chunk_info["start"],
                        "char_end": chunk_info["end"],
                    }
                )

        return chunks

    def _chunk_single_page(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[Dict]:
        """Chunk a single page's text, tracking character positions"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            if end < text_length:
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size * 0.5:
                    chunk = text[start : start + break_point + 1]
                    end = start + break_point + 1

            chunk_text = chunk.strip()
            if chunk_text and len(chunk_text) >= 20:
                chunks.append(
                    {"text": chunk_text, "start": start, "end": min(end, text_length)}
                )

            start = end - chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Load all PDFs with page-by-page tracking

        Returns list of:
        - chunks: list of chunk dicts with text, page_number, char_start, char_end
        - metadata: source info, filename, filepath, total_pages, html_path
        """
        all_chunks = []

        for source_dir in self.data_dir.iterdir():
            if not source_dir.is_dir():
                continue
            if source_dir.name == "vectorstore":
                continue

            source_name = source_dir.name

            for pdf_file in source_dir.glob("*.pdf"):
                print(f"Processing: {pdf_file}")

                try:
                    pages = self.load_pdf_with_pages(pdf_file)
                    total_pages = len(pages)

                    if self.generate_html:
                        html_result = self.html_converter.convert_pdf_to_html(
                            pdf_file, source_name
                        )
                        html_path = html_result["full_html_path"]
                        print(f"  Generated HTML: {html_path}")
                    else:
                        html_path = None

                    page_chunks = self.chunk_pages(pages)

                    for i, chunk in enumerate(page_chunks):
                        chunk_metadata = {
                            "source": source_name,
                            "filename": pdf_file.name,
                            "filepath": str(pdf_file),
                            "html_path": html_path,
                            "page_number": chunk["page_number"],
                            "char_start": chunk["char_start"],
                            "char_end": chunk["char_end"],
                            "chunk_index": i,
                            "total_chunks": len(page_chunks),
                            "total_pages": total_pages,
                        }

                        all_chunks.append(
                            {"text": chunk["text"], "metadata": chunk_metadata}
                        )

                    print(
                        f"  {pdf_file.name}: {len(page_chunks)} chunks from {total_pages} pages"
                    )

                except Exception as e:
                    print(f"Failed to load {pdf_file}: {e}")

        return all_chunks

    @staticmethod
    def generate_id(text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID for a document chunk"""
        content = f"{text}_{metadata.get('source', '')}_{metadata.get('filename', '')}_{metadata.get('chunk_index', '')}"
        return hashlib.md5(content.encode()).hexdigest()
