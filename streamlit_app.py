"""Evidence dashboard for the confirmed synthetic wafer classifier."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
EVIDENCE = json.loads(
    (ROOT / "evidence" / "public_synthetic_evaluation.json").read_text(
        encoding="utf-8"
    )
)
METADATA = json.loads(
    (ROOT / "models" / "public_synthetic_resnet18_v1.json").read_text(
        encoding="utf-8"
    )
)
ASSETS = ROOT / "docs" / "assets"


def percentage(value: float) -> str:
    percent = (Decimal(str(value)) * Decimal("100")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return f"{percent}%"

st.set_page_config(page_title="Wafer Pattern Evidence", page_icon="WP", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=Manrope:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: "Manrope", sans-serif; }
    .stApp { background: #f5f8f7; color: #1e2927; }
    h1, h2, h3 { color: #17324d; letter-spacing: 0; }
    h1 { font-size: 2rem !important; }
    [data-testid="stMetric"] { border-top: 3px solid #18745a; padding-top: .7rem; }
    [data-testid="stMetricValue"] { color: #17324d; font-family: "IBM Plex Mono", monospace; }
    .contract { border-top: 1px solid #d2ddda; border-bottom: 1px solid #d2ddda; padding: .7rem 0; color: #60716d; }
    .boundary { border-left: 4px solid #e0ad3b; padding: .8rem 1rem; background: #fff9e8; }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid #d2ddda; }
    .stTabs [data-baseweb="tab"] { border-radius: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

metrics = EVIDENCE["confirmation_metrics"]
st.title("Wafer Pattern Evidence Room")
st.markdown(
    f'<div class="contract"><strong>{METADATA["model_version"]}</strong> &nbsp;|&nbsp; '
    f'SHA-256 <code>{METADATA["onnx_sha256"][:16]}...</code> &nbsp;|&nbsp; '
    'Grouped synthetic confirmation</div>',
    unsafe_allow_html=True,
)

confirmation_tab, design_tab, classes_tab, runtime_tab = st.tabs(
    ["Confirmation", "Selection design", "Class analysis", "Runtime"]
)

with confirmation_tab:
    columns = st.columns(6)
    columns[0].metric("Samples", f'{metrics["samples"]:,}')
    columns[1].metric("Accuracy", percentage(metrics["accuracy"]))
    columns[2].metric("Macro F1", f'{metrics["macro_f1"]:.4f}')
    columns[3].metric("MCC", f'{metrics["mcc"]:.4f}')
    columns[4].metric("Min recall", percentage(metrics["minimum_class_recall"]))
    columns[5].metric("ECE", f'{metrics["top_label_ece"]:.4f}')
    st.image(str(ASSETS / "confirmation_metrics.png"), width="stretch")

with design_tab:
    dataset = EVIDENCE["dataset"]
    columns = st.columns(4)
    columns[0].metric("Train", f'{dataset["train_samples"]:,}')
    columns[1].metric("Validation", f'{dataset["validation_samples"]:,}')
    columns[2].metric("Confirmation", f'{dataset["confirmation_samples"]:,}')
    columns[3].metric("Group overlaps", sum(EVIDENCE["group_overlap"].values()))
    st.image(str(ASSETS / "benchmark_design.png"), width="stretch")

with classes_tab:
    class_rows = [
        {"Class": class_name, **class_metrics}
        for class_name, class_metrics in metrics["per_class"].items()
    ]
    st.dataframe(pd.DataFrame(class_rows), width="stretch", hide_index=True)
    st.image(str(ASSETS / "confirmation_confusion_matrix.png"), width="stretch")

with runtime_tab:
    latency = EVIDENCE["local_onnx_latency"]
    columns = st.columns(5)
    columns[0].metric("P50", f'{latency["p50_ms"]:.2f} ms')
    columns[1].metric("P95", f'{latency["p95_ms"]:.2f} ms')
    columns[2].metric("P99", f'{latency["p99_ms"]:.2f} ms')
    columns[3].metric("Classes", len(METADATA["class_names"]))
    columns[4].metric("Parity error", f'{EVIDENCE["artifacts"]["maximum_pytorch_onnx_logit_error"]:.2e}')
    st.markdown(
        '<div class="boundary">Independent synthetic image classification only. '
        'WM-811K quality, STDF parsing, yield prediction, and production silicon outcomes '
        'are not established.</div>',
        unsafe_allow_html=True,
    )
