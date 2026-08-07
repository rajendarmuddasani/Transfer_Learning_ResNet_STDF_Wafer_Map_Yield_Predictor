"""Native Streamlit contracts for the Project 4 evidence room."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_loads_canonical_views_and_metrics():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Wafer Pattern Evidence Room"
    assert [tab.label for tab in app.tabs] == [
        "Confirmation",
        "Selection design",
        "Class analysis",
        "Runtime",
    ]
    values = {metric.label: metric.value for metric in app.metric}
    assert values["Accuracy"] == "93.63%"
    assert values["Macro F1"] == "0.9361"
    assert values["Min recall"] == "83.00%"
    assert values["Group overlaps"] == "0"
