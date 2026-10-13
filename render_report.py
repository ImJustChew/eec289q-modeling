#!/usr/bin/env python3
"""Render a Markdown homework report to a polished PDF on Windows.

Usage:
    python render_report.py hw1/report.md
    python render_report.py hw2/report.md --output hw2/hw2.pdf --page-size A4

The renderer intentionally has no third-party Markdown dependency.  It supports
the report constructs used by this project (headings, paragraphs, lists,
tables, links, code, LaTeX math, and relative images), then lets Chrome handle
pagination and MathJax handle equation layout.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote


MATHJAX_URL = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
)


def sanitize_tex(formula: str) -> str:
    """Keep common source typos from becoming visible MathJax error text."""
    # The report's gradient equation uses y_pred_i_i, which MathJax parses as
    # two consecutive subscripts. Render the intended indexed prediction while
    # leaving the Markdown source unchanged.
    return formula.replace(r"y_pred_i_i", r"y_{\mathrm{pred},i}")


def find_chrome() -> Path:
    candidates = [
        os.environ.get("CHROME_PATH"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("chrome"),
        shutil.which("msedge"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError(
        "Chrome or Edge was not found. Set CHROME_PATH to a browser executable."
    )


def file_url(path: Path) -> str:
    """Create a file:// URL that works with Windows paths and spaces."""
    return path.resolve().as_uri()


def replace_math(text: str, math_store: list[tuple[str, bool]]) -> str:
    """Replace TeX spans with placeholders before normal HTML escaping."""

    # Display math is checked first, including the common one-line $$...$$ form.
    pattern = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

    def display(match: re.Match[str]) -> str:
        index = len(math_store)
        math_store.append((sanitize_tex(match.group(1).strip()), True))
        return f"\x00MATH{index}\x00"

    text = pattern.sub(display, text)

    inline_patterns = [
        re.compile(r"\\\((.+?)\\\)", re.DOTALL),
        re.compile(r"\$(?!\$)(.+?)(?<!\$)\$", re.DOTALL),
    ]
    for pattern in inline_patterns:

        def inline(match: re.Match[str]) -> str:
            index = len(math_store)
            math_store.append((sanitize_tex(match.group(1).strip()), False))
            return f"\x00MATH{index}\x00"

        text = pattern.sub(inline, text)
    return text


def render_inline(text: str, math_store: list[tuple[str, bool]]) -> str:
    text = replace_math(text, math_store)
    escaped = html.escape(text, quote=False)

    # Protect inline code before applying emphasis markup.
    code_store: list[str] = []

    def save_code(match: re.Match[str]) -> str:
        index = len(code_store)
        code_store.append(f"<code>{html.escape(match.group(1), quote=False)}</code>")
        return f"\x00CODE{index}\x00"

    escaped = re.sub(r"`([^`]+)`", save_code, escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^\s)]+)(?:\s+['\"]([^'\"]*)['\"])?\)",
                     lambda m: (
                         f'<a href="{html.escape(m.group(2), quote=True)}">'
                         f"{m.group(1)}</a>"
                     ), escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)

    for index, value in enumerate(code_store):
        escaped = escaped.replace(f"\x00CODE{index}\x00", value)

    for index, (formula, display) in enumerate(math_store):
        marker = f"\x00MATH{index}\x00"
        delimiters = (r"\[" + formula + r"\]" if display else r"\(" + formula + r"\)")
        escaped = escaped.replace(marker, delimiters)
    return escaped


def split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return len(cells) > 0 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def is_image_line(line: str) -> re.Match[str] | None:
    return re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line.strip())


def markdown_to_html(markdown: str, source_dir: Path) -> tuple[str, list[Path]]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    math_store: list[tuple[str, bool]] = []
    assets: list[Path] = []
    index = 0

    def paragraph(buffer: list[str]) -> None:
        if buffer:
            text = " ".join(part.strip() for part in buffer)
            output.append(f"<p>{render_inline(text, math_store)}</p>")
            buffer.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", stripped)
        if heading:
            level = len(heading.group(1))
            output.append(f"<h{level}>{render_inline(heading.group(2), math_store)}</h{level}>")
            index += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            language_class = f' class="language-{html.escape(language)}"' if language else ""
            output.append(f"<pre><code{language_class}>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        if stripped == "$$":
            index += 1
            formula_lines: list[str] = []
            while index < len(lines) and lines[index].strip() != "$$":
                formula_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            formula = html.escape(sanitize_tex(chr(10).join(formula_lines).strip()), quote=False)
            output.append(f"<div class=\"math-block\">\\[{formula}\\]</div>")
            continue

        image = is_image_line(stripped)
        if image:
            alt, raw_path = image.group(1), image.group(2)
            relative_path = raw_path.replace("\\", "/")
            asset_path = (source_dir / Path(relative_path)).resolve()
            if not asset_path.is_file():
                raise FileNotFoundError(f"Image referenced by report was not found: {asset_path}")
            assets.append(asset_path)
            output.append(
                '<figure class="report-figure">'
                f'<img src="{html.escape(relative_path, quote=True)}" '
                f'alt="{html.escape(alt, quote=True)}">'
                f'<figcaption>{render_inline(alt, math_store)}</figcaption>'
                "</figure>"
            )
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines) and is_table_separator(lines[index + 1]):
            headers = split_table_row(stripped)
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(split_table_row(lines[index]))
                index += 1
            output.append('<table><thead><tr>')
            output.extend(f"<th>{render_inline(cell, math_store)}</th>" for cell in headers)
            output.append("</tr></thead><tbody>")
            for row in rows:
                output.append("<tr>")
                for cell in row[:len(headers)]:
                    output.append(f"<td>{render_inline(cell, math_store)}</td>")
                output.append("</tr>")
            output.append("</tbody></table>")
            continue

        if re.match(r"^[-*+]\s+", stripped):
            items: list[str] = []
            while index < len(lines):
                match = re.match(r"^[-*+]\s+(.+)$", lines[index].strip())
                if not match:
                    break
                items.append(f"<li>{render_inline(match.group(1), math_store)}</li>")
                index += 1
            output.append("<ul>" + "".join(items) + "</ul>")
            continue

        if re.match(r"^\d+[.)]\s+", stripped):
            items = []
            while index < len(lines):
                match = re.match(r"^\d+[.)]\s+(.+)$", lines[index].strip())
                if not match:
                    break
                items.append(f"<li>{render_inline(match.group(1), math_store)}</li>")
                index += 1
            output.append("<ol>" + "".join(items) + "</ol>")
            continue

        # A normal paragraph continues until a structural Markdown line or blank line.
        buffer = [stripped]
        index += 1
        while index < len(lines) and lines[index].strip():
            next_line = lines[index].strip()
            if (
                re.match(r"^#{1,6}\s+", next_line)
                or next_line.startswith("```")
                or next_line == "$$"
                or is_image_line(next_line)
                or next_line.startswith("|")
                or re.match(r"^[-*+]\s+", next_line)
                or re.match(r"^\d+[.)]\s+", next_line)
            ):
                break
            buffer.append(next_line)
            index += 1
        paragraph(buffer)

    return "\n".join(output), assets


def build_document(body: str, title: str, page_size: str, mathjax_url: str) -> str:
    page_size = page_size.upper()
    if page_size not in {"LETTER", "A4"}:
        raise ValueError("page size must be LETTER or A4")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  @page {{ size: {page_size}; margin: 0.68in 0.72in 0.72in; }}
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  html {{ background: #fff; }}
  body {{
    color: #202124;
    background: #fff;
    font-family: "Times New Roman", Times, serif;
    font-size: 10.5pt;
    line-height: 1.46;
    margin: 0 auto;
    max-width: 7.1in;
  }}
  h1, h2, h3, h4 {{ color: #111827; line-height: 1.2; break-after: avoid; }}
  h1 {{ font-size: 20pt; margin: 0 0 0.28in; }}
  h2 {{ border-bottom: 1px solid #d1d5db; font-size: 15pt; margin: 0.28in 0 0.12in; padding-bottom: 0.04in; }}
  h3 {{ font-size: 12.5pt; margin: 0.22in 0 0.08in; }}
  p {{ margin: 0 0 0.11in; orphans: 3; widows: 3; }}
  ul, ol {{ margin: 0.04in 0 0.13in 0.28in; padding-left: 0.18in; }}
  li {{ margin: 0.03in 0; }}
  a {{ color: #145da0; text-decoration: none; }}
  code {{ background: #f3f4f6; border-radius: 3px; font-family: "Courier New", monospace; font-size: 0.9em; padding: 0.02em 0.22em; }}
  pre {{ background: #f3f4f6; border-left: 3px solid #9ca3af; break-inside: avoid; padding: 0.12in; white-space: pre-wrap; }}
  pre code {{ background: transparent; padding: 0; }}
  table {{ border-collapse: collapse; break-inside: avoid; margin: 0.13in 0 0.18in; width: 100%; }}
  thead {{ display: table-header-group; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 0.055in 0.08in; text-align: left; }}
  th {{ background: #eef2f7; color: #111827; font-weight: 650; }}
  tr {{ break-inside: avoid; }}
  .report-figure {{ break-inside: avoid; margin: 0.16in auto 0.2in; page-break-inside: avoid; text-align: center; }}
  .report-figure img {{ display: block; height: auto; margin: 0 auto; max-height: 8.1in; max-width: 100%; width: auto; }}
  .report-figure figcaption {{ color: #4b5563; font-size: 8.8pt; line-height: 1.25; margin-top: 0.055in; }}
  .math-block {{ margin: 0.14in 0; overflow: visible; text-align: center; }}
  mjx-container[display="true"] {{ margin: 0.14in 0 !important; overflow: visible !important; }}
  @media print {{ a {{ color: inherit; }} }}
</style>
<script>
  window.MathJax = {{
    tex: {{ inlineMath: [['\\\\(', '\\\\)'], ['$', '$']], displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']] }},
    svg: {{ fontCache: 'global' }},
    startup: {{
      pageReady: () => MathJax.startup.defaultPageReady().then(() => {{
        document.documentElement.classList.add('math-ready');
      }})
    }}
  }};
</script>
<script defer src="{html.escape(mathjax_url, quote=True)}"></script>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Markdown report to render")
    parser.add_argument("-o", "--output", type=Path, help="PDF path (default: beside source)")
    parser.add_argument("--page-size", choices=["letter", "a4", "LETTER", "A4"], default="letter")
    parser.add_argument("--mathjax-url", default=MATHJAX_URL, help="MathJax URL, or a local file URL/path")
    parser.add_argument("--keep-html", action="store_true", help="Keep the temporary HTML next to the output PDF")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_file():
        parser.error(f"source file does not exist: {source}")
    output = (args.output or source.with_suffix(".pdf")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    body, assets = markdown_to_html(source.read_text(encoding="utf-8"), source.parent)
    mathjax_url = args.mathjax_url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", mathjax_url):
        mathjax_url = file_url(Path(mathjax_url))
    document = build_document(body, source.stem, args.page_size, mathjax_url)

    chrome = find_chrome()
    temp_html: Path | None = None
    try:
        if args.keep_html:
            temp_html = output.with_name(output.stem + ".render.html")
            temp_html.write_text(document, encoding="utf-8")
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".html", prefix="report-render-", dir=source.parent,
                encoding="utf-8", delete=False
            ) as handle:
                handle.write(document)
                temp_html = Path(handle.name)

        if output.exists():
            output.unlink()
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            "--virtual-time-budget=12000",
            "--run-all-compositor-stages-before-draw",
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            f"--print-to-pdf={output}",
            file_url(temp_html),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not output.is_file():
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Chrome failed to render the report: {details}")
    finally:
        if temp_html and temp_html.exists() and not args.keep_html:
            temp_html.unlink()

    print(f"Rendered {source} -> {output}")
    print(f"Validated {len(assets)} local figure reference(s); page size: {args.page_size.upper()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
