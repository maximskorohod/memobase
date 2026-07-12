"""Small binary-fixture builders for memobase's extract.py tests.

Both builders produce genuinely valid, minimal files for their format (not
mocks/stubs) so extract.py's real parsing path (pdfplumber/pypdf, mammoth) is
exercised end-to-end, not just its degrade-gracefully branches.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def make_minimal_pdf(path: Path, *, text: str = "Hello World") -> Path:
    """Write a minimal, hand-built, valid single-page PDF containing *text*
    as real extractable content (Helvetica/WinAnsi — ASCII only, base-14
    fonts have no Cyrillic glyphs without embedding a font subset, which is
    out of scope for a tiny test fixture). Byte offsets for the xref table
    are computed from the actual bytes written, not hardcoded, so this stays
    correct if *text* changes length.
    """
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 300 200] /Contents 5 0 R >>"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    stream_body = f"BT /F1 18 Tf 10 100 Td ({text}) Tj ET".encode("ascii")
    objects.append(
        b"<< /Length " + str(len(stream_body)).encode("ascii") + b" >>\nstream\n"
        + stream_body + b"\nendstream"
    )

    header = b"%PDF-1.4\n"
    body_parts: list[bytes] = []
    offsets: list[int] = [0]  # object 0 is the free-list head, offset unused
    pos = len(header)
    for i, obj in enumerate(objects, start=1):
        offsets.append(pos)
        chunk = f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
        body_parts.append(chunk)
        pos += len(chunk)

    xref_offset = pos
    n = len(objects) + 1
    xref_lines = [f"xref\n0 {n}\n".encode("ascii"), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref_lines.append(f"{off:010d} 00000 n \n".encode("ascii"))
    trailer = (
        f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("ascii")
    )

    pdf_bytes = header + b"".join(body_parts) + b"".join(xref_lines) + trailer
    path.write_bytes(pdf_bytes)
    return path


def make_minimal_docx(path: Path, *, heading: str = "Заголовок",
                       paragraph: str = "Первый абзац текста на русском языке для проверки экстрактора.") -> Path:
    """Write a minimal, valid .docx (an OOXML zip) with one heading-styled
    paragraph and one plain paragraph, both in Russian — real bytes mammoth
    can open, not a stub."""
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>{heading}</w:t></w:r></w:p>'
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("word/document.xml", document_xml)
    return path
