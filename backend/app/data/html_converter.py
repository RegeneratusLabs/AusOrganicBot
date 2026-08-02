import subprocess
from pathlib import Path
from typing import Dict, List
import pdfplumber


class HTMLConverter:
    """Convert PDFs to clean HTML with page tracking"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def convert_pdf_to_html(self, pdf_path: Path, source: str) -> Dict:
        """
        Convert PDF to clean HTML format
        Returns: { html_path, pages_data, total_pages }
        """
        html_dir = pdf_path.parent / "html"
        html_dir.mkdir(exist_ok=True)

        pages_data = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages_data.append(
                    {"page_number": page_num, "text": text, "char_count": len(text)}
                )

        full_html_path = self._generate_clean_html(
            pdf_path, html_dir, pages_data, source
        )

        return {
            "html_path": str(html_dir),
            "full_html_path": str(full_html_path),
            "pages_data": pages_data,
            "total_pages": total_pages,
        }

    def _generate_clean_html(
        self, pdf_path: Path, html_dir: Path, pages_data: List[Dict], source: str
    ) -> Path:
        """Generate clean HTML with page divs"""

        filename = pdf_path.stem
        full_html_path = html_dir / "full.html"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{filename}</title>
    <style>
        body {{
            font-family: 'Geist', -apple-system, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            background: #fff;
            color: #1e293b;
        }}
        .document-header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #6B9080;
        }}
        .document-header h1 {{
            font-size: 1.5rem;
            color: #1E3A5F;
            margin-bottom: 0.5rem;
        }}
        .document-header .source {{
            font-size: 0.875rem;
            color: #6B9080;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        .page {{
            margin-bottom: 3rem;
            padding: 1.5rem;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        .page-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }}
        .page-number {{
            font-weight: 600;
            color: #6B9080;
            font-size: 0.875rem;
        }}
        .page-content {{
            white-space: pre-wrap;
            font-size: 0.9375rem;
        }}
        .highlight {{
            background: #fef3c7;
            padding: 0.1em 0.2em;
            border-radius: 3px;
            box-shadow: 0 0 0 2px #f59e0b40;
        }}
    </style>
</head>
<body>
    <div class="document-header">
        <h1>{filename}</h1>
        <div class="source">{source.upper()}</div>
    </div>
"""

        for page in pages_data:
            escaped_text = self._escape_html(page["text"])
            html_content += f"""
    <div class="page" data-page="{page["page_number"]}" id="page-{page["page_number"]}">
        <div class="page-header">
            <span class="page-number">Page {page["page_number"]}</span>
        </div>
        <div class="page-content">{escaped_text}</div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        with open(full_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        for page in pages_data:
            page_html_path = html_dir / f"page-{page['page_number']}.html"
            page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Page {page["page_number"]} - {filename}</title>
    <style>
        body {{ font-family: 'Geist', sans-serif; padding: 2rem; line-height: 1.6; }}
        .page-content {{ white-space: pre-wrap; }}
        .highlight {{ background: #fef3c7; padding: 0.1em 0.2em; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="page-content">{self._escape_html(page["text"])}</div>
</body>
</html>
"""
            with open(page_html_path, "w", encoding="utf-8") as f:
                f.write(page_html)

        return full_html_path

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def check_html_exists(self, source: str, filename: str) -> bool:
        """Check if HTML version already exists"""
        html_path = self.data_dir / source / "html" / "full.html"
        return html_path.exists()

    def get_html_path(self, source: str) -> Path:
        """Get HTML directory path for a source"""
        return self.data_dir / source / "html"
