#!/usr/bin/env python3
"""
Download organic certification standards from official sources.

Sources:
- NASAA: https://nasaaorganic.org.au
- ACO: https://aco.net.au
- SXO: https://sxo.com.au
- National Standard: agriculture.gov.au
"""

import httpx
from pathlib import Path
from bs4 import BeautifulSoup
import re


def download_nasaa_standards(output_dir: Path):
    """Download NASAA Organic & Biodynamic Standard"""
    print("\nDownloading NASAA standards...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # NASAA standards page
    url = "https://nasaaorganic.org.au/certification/standards/"

    try:
        response = httpx.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find PDF links
        pdf_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf") or "pdf" in href.lower():
                pdf_links.append(href)

        print(f"  Found {len(pdf_links)} PDF links")

        # Download PDFs
        for i, pdf_url in enumerate(pdf_links[:5]):  # Limit to 5 PDFs
            if not pdf_url.startswith("http"):
                pdf_url = f"https://nasaaorganic.org.au{pdf_url}"

            print(f"  Downloading: {pdf_url}")

            try:
                pdf_response = httpx.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()

                # Extract filename from URL
                filename = pdf_url.split("/")[-1]
                if not filename.endswith(".pdf"):
                    filename = f"nasaa_standard_{i}.pdf"

                output_path = output_dir / filename
                output_path.write_bytes(pdf_response.content)
                print(f"    ✓ Saved: {output_path}")

            except Exception as e:
                print(f"    ✗ Failed: {e}")

    except Exception as e:
        print(f"  ✗ Failed to access NASAA: {e}")
        print(
            "  Manual download required from: https://nasaaorganic.org.au/certification/standards/"
        )


def download_aco_standards(output_dir: Path):
    """Download ACO Standards"""
    print("\nDownloading ACO standards...")

    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://aco.net.au/standards/"

    try:
        response = httpx.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        pdf_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf") or "pdf" in href.lower():
                pdf_links.append(href)

        print(f"  Found {len(pdf_links)} PDF links")

        for i, pdf_url in enumerate(pdf_links[:5]):
            if not pdf_url.startswith("http"):
                pdf_url = f"https://aco.net.au{pdf_url}"

            print(f"  Downloading: {pdf_url}")

            try:
                pdf_response = httpx.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()

                filename = pdf_url.split("/")[-1]
                if not filename.endswith(".pdf"):
                    filename = f"aco_standard_{i}.pdf"

                output_path = output_dir / filename
                output_path.write_bytes(pdf_response.content)
                print(f"    ✓ Saved: {output_path}")

            except Exception as e:
                print(f"    ✗ Failed: {e}")

    except Exception as e:
        print(f"  ✗ Failed to access ACO: {e}")
        print("  Manual download required from: https://aco.net.au/standards/")


def download_sxo_standards(output_dir: Path):
    """Download SXO Certification Requirements"""
    print("\nDownloading SXO standards...")

    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://sxo.com.au/certification/"

    try:
        response = httpx.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        pdf_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf") or "pdf" in href.lower():
                pdf_links.append(href)

        print(f"  Found {len(pdf_links)} PDF links")

        for i, pdf_url in enumerate(pdf_links[:5]):
            if not pdf_url.startswith("http"):
                pdf_url = f"https://sxo.com.au{pdf_url}"

            print(f"  Downloading: {pdf_url}")

            try:
                pdf_response = httpx.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()

                filename = pdf_url.split("/")[-1]
                if not filename.endswith(".pdf"):
                    filename = f"sxo_standard_{i}.pdf"

                output_path = output_dir / filename
                output_path.write_bytes(pdf_response.content)
                print(f"    ✓ Saved: {output_path}")

            except Exception as e:
                print(f"    ✗ Failed: {e}")

    except Exception as e:
        print(f"  ✗ Failed to access SXO: {e}")
        print("  Manual download required from: https://sxo.com.au")


def download_national_standard(output_dir: Path):
    """Download National Standard for Organic and Biodynamic Produce"""
    print("\nDownloading National Standard...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Department of Agriculture
    url = "https://www.agriculture.gov.au/organic-biodynamic/standards"

    try:
        response = httpx.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        pdf_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf") or "pdf" in href.lower():
                pdf_links.append(href)

        print(f"  Found {len(pdf_links)} PDF links")

        for i, pdf_url in enumerate(pdf_links[:5]):
            if not pdf_url.startswith("http"):
                pdf_url = f"https://www.agriculture.gov.au{pdf_url}"

            print(f"  Downloading: {pdf_url}")

            try:
                pdf_response = httpx.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()

                filename = pdf_url.split("/")[-1]
                if not filename.endswith(".pdf"):
                    filename = f"national_standard_{i}.pdf"

                output_path = output_dir / filename
                output_path.write_bytes(pdf_response.content)
                print(f"    ✓ Saved: {output_path}")

            except Exception as e:
                print(f"    ✗ Failed: {e}")

    except Exception as e:
        print(f"  ✗ Failed to access agriculture.gov.au: {e}")
        print(
            "  Manual download required from: https://www.agriculture.gov.au/organic-biodynamic"
        )


def main():
    """Main download function"""
    print("=" * 60)
    print("AusOrganicBot - Standards Document Downloader")
    print("=" * 60)

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    download_nasaa_standards(data_dir / "nasaa")
    download_aco_standards(data_dir / "aco")
    download_sxo_standards(data_dir / "sxo")
    download_national_standard(data_dir / "national")

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Documents saved to: {data_dir}")
    print("\nNext step: Run 'python scripts/ingest.py' to process documents")
    print("=" * 60)


if __name__ == "__main__":
    main()
