"""
streamlit_app/app.py
--------------------
Interactive web UI for the Emotion Classifier + Crisis Detector.

Run locally:
    cd <project_root>
    streamlit run streamlit_app/app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.models.llm_classifier import LLMClassifier, GOEMOTIONS_LABELS
from src.crisis_detection import rule_based_crisis, integrated_analysis
from utils.helpers import labels_to_groups, dominant_group, EMOTION_HIERARCHY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense — Emotion & Crisis Analyser",
    page_icon="🧠",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .emotion-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .crisis-box {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .safe-box {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 EmotiSense</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hierarchical Multi-Label Emotion Classification & Crisis Detection</div>',
    unsafe_allow_html=True
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    use_api = st.toggle("Use Anthropic API", value=False,
                         help="Set ANTHROPIC_API_KEY env var to enable real API calls.")
    threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.3, 0.05)
    top_k = st.slider("Max labels to return", 1, 5, 3)
    show_probs = st.checkbox("Show probability scores", value=True)

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "This demo runs an LLM-based simulation classifier (no GPU required). "
        "Switch on the API toggle and set your `ANTHROPIC_API_KEY` for real LLM inference."
    )
    st.markdown("**Dataset**: Google GoEmotions (28 emotions)")
    st.markdown("**Crisis lexicon**: 40+ crisis phrases")

# ── Load classifier ───────────────────────────────────────────────────────────
@st.cache_resource
def get_classifier(api: bool):
    return LLMClassifier(use_api=api, label_names=GOEMOTIONS_LABELS)

clf = get_classifier(use_api and bool(os.getenv("ANTHROPIC_API_KEY", "")))

# ── Emotion colour map ────────────────────────────────────────────────────────
GROUP_COLOURS = {
    "positive":  "#10B981",  # green
    "negative":  "#EF4444",  # red
    "ambiguous": "#F59E0B",  # amber
    "neutral":   "#6B7280",  # gray
}
EMOTION_TO_GROUP = {
    e: g
    for g, emotions in EMOTION_HIERARCHY.items()
    for e in emotions
}

def emotion_tag_html(emotion: str) -> str:
    colour = GROUP_COLOURS.get(EMOTION_TO_GROUP.get(emotion, "neutral"), "#6B7280")
    return (
        f'<span class="emotion-tag" '
        f'style="background-color:{colour}22; color:{colour}; '
        f'border: 1px solid {colour};">{emotion}</span>'
    )

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyse Text", "📊 Batch Analysis", "📈 Model Comparison"])

# ─────────────── Tab 1: Single text ──────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Enter Text to Analyse")
        example_texts = [
            "I am absolutely thrilled and grateful for this opportunity!",
            "I feel so hopeless… I can't see a reason to keep going.",
            "The movie was okay I guess, nothing special.",
            "I'm scared about what might happen next, but also curious.",
            "I want to end my life. Nobody would miss me anyway.",
        ]
        selected = st.selectbox("📌 Try an example:", ["(type your own)"] + example_texts)
        if selected != "(type your own)":
            default_text = selected
        else:
            default_text = ""

        user_text = st.text_area(
            "Your text:",
            value=default_text,
            height=140,
            placeholder="Type or paste a social media comment, Reddit post, etc.",
        )
        analyse_btn = st.button("🔬 Analyse", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Emotion Groups")
        for group, emotions in EMOTION_HIERARCHY.items():
            colour = GROUP_COLOURS[group]
            with st.expander(f"{'🟢' if group=='positive' else '🔴' if group=='negative' else '🟡' if group=='ambiguous' else '⚪'} {group.capitalize()} ({len(emotions)})"):
                st.write(", ".join(emotions))

    if analyse_btn and user_text.strip():
        with st.spinner("Analysing …"):
            labels = clf.predict_labels([user_text], top_k=top_k)[0]
            groups = labels_to_groups(labels)
            dom    = dominant_group(labels)

            crisis_label, crisis_score, triggers = rule_based_crisis(user_text)
            result = integrated_analysis(user_text, labels)

        st.divider()
        st.subheader("📊 Results")

        # Emotion tags
        tags_html = " ".join(emotion_tag_html(e) for e in labels)
        st.markdown(f"**Detected Emotions:** {tags_html}", unsafe_allow_html=True)
        st.markdown(f"**Dominant Group:** `{dom.upper()}`")

        # Hierarchy breakdown
        group_counts = {}
        for lbl in labels:
            g = EMOTION_TO_GROUP.get(lbl, "neutral")
            group_counts[g] = group_counts.get(g, 0) + 1

        if group_counts:
            fig, ax = plt.subplots(figsize=(4, 3))
            colours = [GROUP_COLOURS.get(g, "#ccc") for g in group_counts]
            ax.pie(group_counts.values(), labels=group_counts.keys(),
                   colors=colours, autopct="%1.0f%%", startangle=90)
            ax.set_title("Emotion Group Distribution")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        # Crisis box
        if result["crisis"]:
            st.markdown(
                f'<div class="crisis-box">'
                f'<b>⚠️ Crisis Signal Detected</b><br>'
                f'Severity score: {crisis_score:.2f}<br>'
                f'Triggers: {", ".join(triggers) if triggers else "pattern match"}<br><br>'
                f'{result["recommendation"]}'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="safe-box">'
                f'<b>✅ No Crisis Indicators</b><br>'
                f'{result["recommendation"]}'
                f'</div>', unsafe_allow_html=True
            )

# ─────────────── Tab 2: Batch ─────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Analysis")
    st.markdown("Paste multiple texts (one per line) for bulk emotion classification.")

    batch_input = st.text_area(
        "Texts (one per line):",
        height=200,
        placeholder="I love this!\nThis is the worst day ever.\nFeeling okay I guess.",
    )
    run_batch = st.button("▶️ Run Batch", type="primary")

    if run_batch and batch_input.strip():
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        with st.spinner(f"Processing {len(lines)} texts …"):
            all_labels = clf.predict_labels(lines, top_k=top_k)
            rows = []
            for text, labels in zip(lines, all_labels):
                cr_label, cr_score, triggers = rule_based_crisis(text)
                rows.append({
                    "Text":    text[:80] + ("…" if len(text) > 80 else ""),
                    "Emotions":   ", ".join(labels),
                    "Group":      dominant_group(labels),
                    "Crisis":     "⚠️ YES" if cr_label else "✅ No",
                    "Severity":   f"{cr_score:.2f}",
                })

        df_results = pd.DataFrame(rows)
        st.dataframe(df_results, use_container_width=True)

        # Label frequency chart
        all_emotion_labels = [e for labels in all_labels for e in labels]
        from collections import Counter
        freq = Counter(all_emotion_labels).most_common(15)
        if freq:
            names, counts = zip(*freq)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            colours2 = [GROUP_COLOURS.get(EMOTION_TO_GROUP.get(n, "neutral"), "#999") for n in names]
            ax2.bar(names, counts, color=colours2)
            ax2.set_title("Emotion Frequency in Batch")
            ax2.set_xlabel("Emotion")
            ax2.set_ylabel("Count")
            ax2.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # Download
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "emotion_results.csv", "text/csv")

# ─────────────── Tab 3: Model Comparison ─────────────────────────────────────
with tab3:
    st.subheader("📈 Model Performance Comparison")
    st.markdown("Illustrative benchmark results on GoEmotions test set:")

    comparison_data = {
        "Model": ["Logistic Regression", "Linear SVM", "BiLSTM", "BERT", "LLM (Simulation)"],
        "F1 (micro)":    [0.42, 0.44, 0.51, 0.62, 0.38],
        "F1 (macro)":    [0.31, 0.33, 0.42, 0.55, 0.27],
        "Precision":     [0.48, 0.50, 0.56, 0.65, 0.44],
        "Recall":        [0.38, 0.39, 0.47, 0.60, 0.34],
        "Training Time": ["~2 min", "~5 min", "~25 min", "~90 min", "~1 min"],
        "Requires GPU":  ["No", "No", "Optional", "Recommended", "No"],
    }
    df_cmp = pd.DataFrame(comparison_data).set_index("Model")
    st.dataframe(df_cmp, use_container_width=True)

    # Bar chart
    numeric_cols = ["F1 (micro)", "F1 (macro)", "Precision", "Recall"]
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_cmp))
    w = 0.2
    for i, col in enumerate(numeric_cols):
        ax3.bar(x + i * w, df_cmp[col], width=w, label=col)
    ax3.set_xticks(x + w * 1.5)
    ax3.set_xticklabels(df_cmp.index, rotation=20, ha="right")
    ax3.set_title("Model Comparison – Key Metrics")
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 0.9)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("""
    **Key Insights:**
    - **BERT** achieves the highest performance across all metrics, benefiting from
      pre-trained contextual representations.
    - **BiLSTM** provides a strong deep-learning baseline with significantly lower
      resource requirements.
    - **SVM** slightly outperforms Logistic Regression and remains competitive for
      resource-constrained deployments.
    - **LLM simulation** is rule-based and has lower recall but near-instant inference
      with no training required.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>EmotiSense · Built with GoEmotions dataset · "
    "For research purposes only · "
    "If you or someone you know is in crisis, call 988 (US) or text HOME to 741741.</small></center>",
    unsafe_allow_html=True,
)
