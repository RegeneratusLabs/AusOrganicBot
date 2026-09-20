import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber
from PyPDF2 import PdfReader

from app.data.chunker import chunk_structured
from app.data.html_converter import HTMLConverter
from app.data.manifest import CountryManifest, ManifestSource, load_manifest

logger = logging.getLogger("orgstandardsbot")


class DocumentLoader:
    """Load and process PDF documents for RAG with page tracking.

    Documents live under `data/<country>/<source>/*.pdf`. Edition/status
    metadata comes from the country manifest when present."""

    def __init__(self, data_dir: Path, country: str = "au", generate_html: bool = True):
        self.data_dir = data_dir
        self.country = country
        self.country_dir = data_dir / country
        self.generate_html = generate_html
        self.html_converter = HTMLConverter(data_dir) if generate_html else None
        try:
            self.manifest: Optional[CountryManifest] = load_manifest(data_dir, country)
        except FileNotFoundError:
            self.manifest = None
        except Exception as e:  # malformed manifest — fall back to globbing
            logger.warning("Manifest for '%s' failed to parse: %s", country, e)
            self.manifest = None

    def _manifest_entry(self, pdf_file: Path) -> Optional[ManifestSource]:
        if not self.manifest:
            return None
        rel = str(pdf_file.relative_to(self.country_dir))
        for source in self.manifest.sources:
            if source.local_path and Path(source.local_path) == Path(rel):
                return source
        return None

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

    # ------------------------------------------------------------------
    # Page header / footer stripping
    # ------------------------------------------------------------------

    # A line repeated on at least this fraction of a document's pages is a
    # running header/footer, not content. Kept relative so short documents are
    # not stripped by accident.
    HEADER_PAGE_RATIO = 0.5
    HEADER_MIN_PAGES = 3
    HEADER_EDGE_LINES = 3

    @staticmethod
    def _normalise_header(line: str) -> str:
        """Collapse a header line so minor differences still compare equal."""
        return re.sub(r"\s+", " ", line).strip().lower()[:120]

    @classmethod
    def strip_running_headers(cls, pages: List[Dict[str, Any]]) -> int:
        """Remove lines repeated at the top/bottom of most pages, in place.

        Regulation PDFs (EU/UK) and government prints (US CFR) repeat a header
        on every page — for example "14/08/2026, 18:05 Commission Regulation
        (EC) No 889/2008 ...". Left in, that boilerplate dominates the text of
        every chunk and pollutes both embeddings and answers.

        Returns the number of lines removed.
        """
        if len(pages) < cls.HEADER_MIN_PAGES:
            return 0

        counts: Counter = Counter()
        for page in pages:
            lines = [ln for ln in page.get("text", "").splitlines() if ln.strip()]
            for line in set(lines[: cls.HEADER_EDGE_LINES]) | set(lines[-cls.HEADER_EDGE_LINES:]):
                counts[cls._normalise_header(line)] += 1

        threshold = max(cls.HEADER_MIN_PAGES, int(len(pages) * cls.HEADER_PAGE_RATIO))
        repeated = {key for key, n in counts.items() if n >= threshold and key}

        removed = 0
        for page in pages:
            lines = page.get("text", "").splitlines()
            kept = [ln for ln in lines if cls._normalise_header(ln) not in repeated]
            removed += len(lines) - len(kept)
            page["text"] = "\n".join(kept)
            page["char_count"] = len(page["text"])
        return removed

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

        removed = self.strip_running_headers(pages)
        if removed:
            logger.info(
                "Stripped %d repeated header/footer line(s) from %s", removed, pdf_path.name
            )
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

    def chunk_pages_structurally(
        self,
        pages: List[Dict[str, Any]],
        max_chars: int = 900,
        min_chars: int = 40,
    ) -> List[Dict[str, Any]]:
        """Chunk pages along section/clause boundaries (preferred ingester).

        Chunking is done per page, because that is the unit of page-tracking the
        document viewer depends on. Within a page, section headings and numbered
        clauses are detected so that no clause is split and each chunk carries
        the section number it belongs to. See app/data/chunker.py.
        """
        chunks: List[Dict[str, Any]] = []

        for page in pages:
            page_text = page["text"]
            page_num = page["page_number"]
            if not page_text.strip():
                continue

            for chunk in chunk_structured(page_text, max_chars=max_chars, min_chars=min_chars):
                chunks.append(
                    {
                        "text": chunk.embed_text,
                        "page_number": page_num,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "section_number": chunk.section_number,
                        "section_title": chunk.section_title,
                        "clause_numbers": chunk.clause_numbers,
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

    def load_all_documents(
        self,
        only_source: Optional[str] = None,
        only_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all active PDFs for this country with page-by-page tracking.

        `only_source` restricts loading to one source directory (e.g. "national")
        and `only_file` to a single PDF filename; together they let a large
        country be ingested in several shorter runs.

        Returns list of:
        - chunks: list of chunk dicts with text, page_number, char_start, char_end
        - metadata: source info, filename, filepath, total_pages, html_path,
                    country_code, edition
        """
        if not self.country_dir.exists():
            raise FileNotFoundError(
                f"Country directory not found: {self.country_dir}. "
                f"Run: python scripts/add_country.py --country {self.country}"
            )

        all_chunks = []

        for source_dir in sorted(self.country_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            if source_dir.name == "html":
                continue
            if only_source and source_dir.name != only_source:
                continue

            source_name = source_dir.name

            for pdf_file in sorted(source_dir.glob("*.pdf")):
                if only_file and pdf_file.name != only_file:
                    continue
                manifest_entry = self._manifest_entry(pdf_file)
                if manifest_entry and manifest_entry.status == "superseded":
                    print(f"Skipping (superseded): {pdf_file.name}")
                    continue

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

                    page_chunks = self.chunk_pages_structurally(pages)

                    edition = manifest_entry.edition if manifest_entry else pdf_file.stem

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
                            "country_code": self.country,
                            "edition": edition,
                            # Structure-aware chunking metadata — lets the reader
                            # cite a real clause number and lets verification
                            # confirm the citation exists in the excerpt.
                            "section_number": chunk.get("section_number") or "",
                            "section_title": chunk.get("section_title") or "",
                            "clause_numbers": ",".join(chunk.get("clause_numbers") or []),
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
    def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Drop values ChromaDB cannot store.

        Chroma's metadata values must be str/int/float/bool. `None` (for example
        ``html_path`` when HTML generation is disabled) raises
        ``TypeError: Cannot convert Python object to MetadataValue``, so None
        values are omitted rather than written. Callers already use ``.get()``
        for optional fields.
        """
        return {k: v for k, v in metadata.items() if v is not None}

    @staticmethod
    def generate_id(text: str, metadata: Dict[str, Any]) -> str:
        """Generate unique ID for a document chunk"""
        content = f"{text}_{metadata.get('source', '')}_{metadata.get('filename', '')}_{metadata.get('chunk_index', '')}"
        return hashlib.md5(content.encode()).hexdigest()
