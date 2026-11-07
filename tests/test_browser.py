"""Real Chromium tests. Enable with RUN_BROWSER_TESTS=1 and install Playwright Chromium."""

import json
import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import pytest
from playwright.sync_api import Browser, Page, expect, sync_playwright

from ml_core import METRICS, MODEL_FILES, ROOT

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Set RUN_BROWSER_TESTS=1 for Chromium tests"
)


@pytest.fixture(scope="module")
def live_url():
    """Start an isolated Streamlit server and stop only this owned process after testing."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (ROOT / "results" / "browser-server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "app.py"),
                "--server.port",
                str(port),
                "--server.address",
                "127.0.0.1",
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "Streamlit exited before becoming ready. See browser-server.log."
                    )
                try:
                    with urlopen(url + "/_stcore/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise TimeoutError("Streamlit health check did not succeed within 60 seconds.")
            yield url
        finally:
            process.terminate()
            process.wait(timeout=15)


@pytest.fixture(scope="module")
def browser():
    """Launch a real Chromium browser for interactive Streamlit verification."""
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        yield instance
        instance.close()


@pytest.fixture
def page(browser: Browser, live_url: str):
    """Provide an isolated browser page at desktop size."""
    context = browser.new_context(viewport={"width": 1440, "height": 1050}, accept_downloads=True)
    page = context.new_page()
    page.goto(live_url)
    expect(page.get_by_role("heading", name="BeanScope", exact=True)).to_be_visible(timeout=60000)
    expect(page.get_by_role("heading", name="Same test rows, side-by-side")).to_be_visible(
        timeout=60000
    )
    expect(page.get_by_role("button", name="Stop", exact=True)).not_to_be_visible(timeout=60000)
    yield page
    context.close()


def upload_csv(page: Page, name: str, content: bytes) -> None:
    """Use the browser's real file input to upload CSV bytes."""
    page.get_by_text("Upload test CSV", exact=True).click()
    page.locator('input[type="file"]').set_input_files(
        {
            "name": name,
            "mimeType": "text/csv",
            "buffer": content,
        }
    )


def test_given_app_when_switching_all_models_then_metrics_match_offline(page: Page) -> None:
    expected = pd.read_csv(ROOT / "results" / "comparison.csv").set_index("Model")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    for name in MODEL_FILES:
        page.get_by_role("combobox").click()
        page.get_by_role("option", name=name, exact=True).click()
        expect(page.get_by_role("heading", name=f"{name} | Evaluation metrics")).to_be_visible()
        expect(page.get_by_role("button", name="Stop", exact=True)).not_to_be_visible(timeout=60000)
        for metric in METRICS:
            card = page.locator('[data-testid="stMetric"]').filter(
                has=page.locator('[data-testid="stMetricLabel"]').get_by_text(metric, exact=True)
            )
            expect(card.locator('[data-testid="stMetricValue"]')).to_have_text(
                f"{expected.loc[name, metric]:.4f}", timeout=30000
            )
    page.get_by_role("tab", name="Model diagnostics").click()
    expect(page.get_by_role("heading", name="Confusion matrix")).to_be_visible()
    expect(page.get_by_role("heading", name="Classification report")).to_be_visible()
    expect(page.locator('[data-testid="stImage"] img')).to_be_visible(timeout=60000)
    page.get_by_role("tab", name="Compare all models").click()
    page.screenshot(path=str(ROOT / "results" / "app-desktop.png"), full_page=True)

    assert not errors


def test_given_valid_upload_when_evaluated_then_results_and_downloads_work(page: Page) -> None:
    content = (ROOT / "test_data.csv").read_bytes()

    upload_csv(page, "test_data.csv", content)

    expect(page.get_by_role("heading", name="Same test rows, side-by-side")).to_be_visible(
        timeout=60000
    )
    expect(page.get_by_text("Evaluating: test_data.csv", exact=False)).to_be_visible()
    with page.expect_download() as download:
        page.get_by_role("button", name="Download comparison", exact=True).click()
    downloaded = pd.read_csv(download.value.path())
    assert downloaded.shape == (5, 7)
    page.get_by_role("tab", name="Dataset & predictions").click()
    with page.expect_download() as download:
        page.get_by_role("button", name="Download predictions", exact=True).click()
    predictions = pd.read_csv(download.value.path())
    assert len(predictions) == len(pd.read_csv(ROOT / "test_data.csv"))
    assert predictions.columns.tolist() == ["Actual", "Predicted", "Confidence"]


def test_given_invalid_upload_when_evaluated_then_error_and_recovery_work(page: Page) -> None:
    upload_csv(page, "invalid.csv", b"Area,Class\n123,SEKER\n")

    expect(page.get_by_text("Missing columns:", exact=False)).to_be_visible(timeout=30000)
    expect(page.get_by_role("heading", name="Same test rows, side-by-side")).not_to_be_visible()
    page.locator('input[type="file"]').set_input_files(str(ROOT / "test_data.csv"))
    expect(page.get_by_role("heading", name="Same test rows, side-by-side")).to_be_visible(
        timeout=60000
    )
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def test_given_single_class_upload_when_evaluated_then_auc_warning_is_visible(page: Page) -> None:
    frame = pd.read_csv(ROOT / "test_data.csv")
    content = frame[frame["Class"] == "SEKER"].head(5).to_csv(index=False).encode()

    upload_csv(page, "single_class.csv", content)

    expect(page.get_by_text("AUC is undefined", exact=False)).to_be_visible(timeout=60000)
    expect(page.locator('[data-testid="stException"]')).to_have_count(0)


def test_given_mobile_view_when_loaded_then_main_controls_remain_usable(page: Page) -> None:
    page.set_viewport_size({"width": 390, "height": 844})

    expect(page.get_by_role("heading", name="BeanScope", exact=True)).to_be_visible()
    page.get_by_test_id("stExpandSidebarButton").click()
    page.get_by_role("combobox").click()
    page.get_by_role("option", name="KNN", exact=True).click()
    page.get_by_test_id("stSidebarCollapseButton").get_by_role("button").click()
    expect(page.get_by_role("heading", name="KNN | Evaluation metrics")).to_be_visible()
    page.screenshot(path=str(ROOT / "results" / "app-mobile.png"), full_page=True)
    report = {
        "browser": "Chromium",
        "verified": [
            "All five model dropdown selections and all six displayed metrics",
            "Confusion matrix and classification report",
            "Real file upload and comparison/prediction CSV downloads",
            "Invalid upload validation and recovery",
            "Single-class AUC warning",
            "Desktop and mobile rendering",
        ],
        "note": "Local browser evidence only; not BITS Virtual Lab execution proof.",
    }
    (ROOT / "results" / "browser-checks.json").write_text(json.dumps(report, indent=2) + "\n")
