"""
Render a plain-text or markdown file into a simple multi-page PDF using only
the Python standard library.

This is intentionally lightweight so it works in restricted environments where
PDF libraries are unavailable.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from textwrap import wrap

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 48
TOP_MARGIN = 48
BOTTOM_MARGIN = 48
FONT_SIZE = 10
LINE_HEIGHT = 14
MAX_TEXT_WIDTH = 92


def normalize_markdown(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            content = line[hashes:].strip()
            if content:
                line = content.upper() if hashes <= 2 else content
        line = line.replace("`", "")
        line = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", line)
        if not line:
            lines.append("")
            continue
        wrapped = wrap(line, width=MAX_TEXT_WIDTH, replace_whitespace=False, drop_whitespace=False)
        lines.extend(wrapped or [""])
    return lines


def paginate(lines: list[str]) -> list[list[str]]:
    usable_height = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    lines_per_page = usable_height // LINE_HEIGHT
    return [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[]]


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_content_stream(page_lines: list[str], page_no: int, page_count: int) -> bytes:
    y = PAGE_HEIGHT - TOP_MARGIN
    parts = ["BT", f"/F1 {FONT_SIZE} Tf"]
    for line in page_lines:
        parts.append(f"1 0 0 1 {LEFT_MARGIN} {y} Tm ({pdf_escape(line)}) Tj")
        y -= LINE_HEIGHT
    footer = f"Page {page_no} of {page_count}"
    parts.append(f"1 0 0 1 {PAGE_WIDTH - 120} 24 Tm ({footer}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def write_pdf(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8")
    lines = normalize_markdown(text)
    pages = paginate(lines)

    objects: list[bytes] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    content_ids: list[int] = []
    for idx, page in enumerate(pages, start=1):
        stream = build_content_stream(page, idx, len(pages))
        payload = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        content_ids.append(add_object(payload))

    page_ids: list[int] = []
    pages_parent_id_placeholder = len(objects) + len(pages) + 1
    for content_id in content_ids:
        page_payload = (
            f"<< /Type /Page /Parent {pages_parent_id_placeholder} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        page_ids.append(add_object(page_payload))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_object(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("latin-1"))
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    xref_positions: list[int] = [0]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    for idx, obj in enumerate(objects, start=1):
        xref_positions.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for pos in xref_positions[1:]:
        pdf.extend(f"{pos:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    pdf.extend(trailer.encode("latin-1"))
    output_path.write_bytes(pdf)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python tools/render_text_pdf.py <input.md> <output.pdf>")
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    write_pdf(input_path, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
