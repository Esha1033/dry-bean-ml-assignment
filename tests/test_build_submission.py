"""Validate external evidence requirements and the submission PDF content renderer."""

import pytest
from PIL import Image

from build_submission import build_pdf, inline_markdown, readme_flowables, validated_url
from ml_core import ROOT


@pytest.mark.parametrize(
    ("url", "github"),
    [
        ("http://github.com/student/project", True),
        ("https://github.com/student", True),
        ("https://example.com/student/project", True),
        ("https://user:password@github.com/student/project", True),
        ("https://example.com", False),
        ("https://notstreamlit.app", False),
    ],
)
def test_given_invalid_deployment_url_when_validated_then_rejected(url: str, github: bool) -> None:
    with pytest.raises(ValueError):
        validated_url(url, github=github)


def test_given_readme_markup_when_rendered_then_sections_are_preserved() -> None:
    text = "---\ntitle: Test\n---\n## Problem\nA `feature`.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"

    flowables = readme_flowables(text)

    assert len(flowables) >= 4
    assert inline_markdown("<unsafe>") == "&lt;unsafe&gt;"


def test_given_full_readme_when_building_then_pdf_and_links_are_created(
    tmp_path, monkeypatch
) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    image = tmp_path / "synthetic-test-image.png"
    Image.new("RGB", (640, 480), "white").save(image)
    monkeypatch.setattr("build_submission.ROOT", tmp_path)
    output = tmp_path / "test-only.pdf"

    build_pdf(
        "https://github.com/test-fixture/example",
        "https://test-fixture.streamlit.app",
        image,
        output,
    )

    assert output.read_bytes().startswith(b"%PDF-")
    assert output.read_bytes().rstrip().endswith(b"%%EOF")
    updated = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "GITHUB_REPOSITORY_URL" not in updated
    assert "STREAMLIT_APP_URL" not in updated
