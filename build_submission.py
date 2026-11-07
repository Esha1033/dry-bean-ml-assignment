"""Build the required single PDF after real deployment links and lab evidence exist.

Usage: uv run build_submission.py --github-url URL --app-url URL --lab-screenshot PATH
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ml_core import ROOT


def validated_url(value: str, *, github: bool = False) -> str:
    """Require real HTTPS repository and Streamlit Community Cloud URL shapes."""
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Provide a public HTTPS URL without embedded credentials.")
    if github:
        if parsed.hostname != "github.com" or len(parsed.path.strip("/").split("/")) != 2:
            raise ValueError("GitHub URL must identify an owner/repository on github.com.")
    elif not parsed.hostname.endswith(".streamlit.app"):
        raise ValueError("The live app must use a Streamlit Community Cloud .streamlit.app URL.")
    return value


def inline_markdown(text: str) -> str:
    """Render escaped Markdown inline links and code spans for the PDF."""
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2">\1</link>', text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def readme_flowables(text: str) -> list:
    """Preserve all README sections, tables, code blocks, and prose in the PDF."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallCell", fontSize=7, leading=9))
    lines = text.splitlines()
    if lines and lines[0] == "---":
        end = lines.index("---", 1)
        lines = lines[end + 1 :]
    blocks = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            blocks.append(Preformatted("\n".join(code), styles["Code"], maxLineLength=90))
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [part.strip() for part in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(
                        [Paragraph(inline_markdown(cell), styles["SmallCell"]) for cell in cells]
                    )
                index += 1
            table = Table(
                rows, repeatRows=1, hAlign="LEFT", colWidths=[504 / len(rows[0])] * len(rows[0])
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E0EBDF")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBC9BB")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            blocks.extend([table, Spacer(1, 10)])
            continue
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            blocks.append(Paragraph(inline_markdown(line.lstrip("# ")), styles[f"Heading{level}"]))
        elif line.strip():
            blocks.append(Paragraph(inline_markdown(line), styles["BodyText"]))
        else:
            blocks.append(Spacer(1, 5))
        index += 1
    return blocks


def build_pdf(github_url: str, app_url: str, lab_screenshot: Path, output: Path) -> None:
    """Build the final PDF and update README link tokens from the supplied real links."""
    github_url = validated_url(github_url, github=True)
    app_url = validated_url(app_url)
    if not lab_screenshot.is_file():
        raise FileNotFoundError("Provide your actual BITS Virtual Lab execution screenshot.")
    width, height = ImageReader(str(lab_screenshot)).getSize()
    readme_path = ROOT / "README.md"
    readme = (
        readme_path.read_text(encoding="utf-8")
        .replace("GITHUB_REPOSITORY_URL", github_url)
        .replace("STREAMLIT_APP_URL", app_url)
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", fontSize=22, leading=28, alignment=TA_CENTER))
    scale = min(504 / width, 420 / height)
    blocks = [
        Paragraph("Machine Learning Assignment 2", styles["CoverTitle"]),
        Spacer(1, 16),
        Paragraph("Dry Bean Multiclass Classification", styles["Heading2"]),
        Paragraph("1. GitHub repository", styles["Heading2"]),
        Paragraph(
            f'<link href="{escape(github_url)}">{escape(github_url)}</link>', styles["BodyText"]
        ),
        Paragraph("2. Live Streamlit application", styles["Heading2"]),
        Paragraph(f'<link href="{escape(app_url)}">{escape(app_url)}</link>', styles["BodyText"]),
        Paragraph("3. BITS Virtual Lab execution screenshot", styles["Heading2"]),
        Image(str(lab_screenshot), width=width * scale, height=height * scale),
        PageBreak(),
        Paragraph("4. GitHub README content", styles["Heading1"]),
        *readme_flowables(readme),
    ]
    SimpleDocTemplate(str(output), rightMargin=45, leftMargin=45).build(blocks)
    readme_path.write_text(readme, encoding="utf-8")


def main() -> None:
    """Collect real external submission evidence and generate the PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-url", required=True)
    parser.add_argument("--app-url", required=True)
    parser.add_argument("--lab-screenshot", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.pdf")
    args = parser.parse_args()
    build_pdf(args.github_url, args.app_url, args.lab_screenshot, args.output)


if __name__ == "__main__":
    main()
