"""
ML Network Intrusion Detector
==============================
A beginner-friendly Streamlit dashboard that:
  - Trains a Random Forest classifier on a synthetic NSL-KDD-style dataset
  - Lets users enter network traffic features and get a live prediction
  - Shows model accuracy, a confusion matrix, and a traffic distribution chart
  - Works completely out-of-the-box without downloading any external dataset

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML Network Intrusion Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — dark cybersecurity theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: #0d1117;
    color: #c9d1d9;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 2rem !important;
    font-weight: 700;
}

/* ── Section headers ── */
h1 { color: #58a6ff !important; }
h2 { color: #79c0ff !important; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { color: #cae8ff !important; }

/* ── Buttons ── */
div.stButton > button {
    background: #1f6feb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: background .2s;
}
div.stButton > button:hover {
    background: #388bfd;
    color: #fff;
}

/* ── Inputs ── */
input, select, textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border-color: #30363d !important;
    border-radius: 6px !important;
}

/* ── Tables / dataframes ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Alert boxes ── */
.stSuccess { background: #0d2a18 !important; border-left: 4px solid #3fb950; }
.stWarning { background: #2a1f00 !important; border-left: 4px solid #d29922; }
.stError   { background: #2a0d0d !important; border-left: 4px solid #f85149; }
.stInfo    { background: #0d1f2a !important; border-left: 4px solid #58a6ff; }

/* ── Expander ── */
details { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA GENERATION  — synthetic NSL-KDD-style
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Creates a synthetic dataset that mirrors the key numeric features of NSL-KDD.
    Labels: normal, dos, probe, r2l, u2r
    This is clearly labeled as synthetic demo data — no real packets are analyzed.
    """
    rng = np.random.default_rng(seed)

    # Class proportions (reflecting NSL-KDD imbalance)
    class_counts = {
        "normal": int(n_samples * 0.45),
        "dos":    int(n_samples * 0.35),
        "probe":  int(n_samples * 0.12),
        "r2l":    int(n_samples * 0.05),
        "u2r":    int(n_samples * 0.03),
    }

    rows = []
    for label, count in class_counts.items():
        for _ in range(count):
            if label == "normal":
                row = dict(
                    duration      = rng.integers(0, 300),
                    src_bytes     = rng.integers(100, 50000),
                    dst_bytes     = rng.integers(100, 50000),
                    land          = 0,
                    wrong_fragment= 0,
                    urgent        = 0,
                    hot           = rng.integers(0, 5),
                    num_failed_logins=0,
                    logged_in     = 1,
                    count         = rng.integers(1, 30),
                    srv_count     = rng.integers(1, 30),
                    serror_rate   = round(rng.uniform(0, 0.1), 3),
                    rerror_rate   = round(rng.uniform(0, 0.1), 3),
                    same_srv_rate = round(rng.uniform(0.8, 1.0), 3),
                    diff_srv_rate = round(rng.uniform(0.0, 0.2), 3),
                    dst_host_count= rng.integers(50, 255),
                    dst_host_srv_count=rng.integers(20, 255),
                    dst_host_serror_rate=round(rng.uniform(0, 0.05), 3),
                    protocol_type = rng.choice(["tcp", "udp", "icmp"], p=[0.7, 0.2, 0.1]),
                    service       = rng.choice(["http", "ftp", "smtp", "ssh", "dns"]),
                    flag          = rng.choice(["SF", "S0", "REJ"], p=[0.9, 0.05, 0.05]),
                    label         = label,
                )
            elif label == "dos":
                row = dict(
                    duration      = rng.integers(0, 5),
                    src_bytes     = rng.integers(0, 500),
                    dst_bytes     = 0,
                    land          = rng.integers(0, 2),
                    wrong_fragment= rng.integers(0, 4),
                    urgent        = 0,
                    hot           = 0,
                    num_failed_logins=0,
                    logged_in     = 0,
                    count         = rng.integers(200, 512),
                    srv_count     = rng.integers(200, 512),
                    serror_rate   = round(rng.uniform(0.8, 1.0), 3),
                    rerror_rate   = round(rng.uniform(0.0, 0.1), 3),
                    same_srv_rate = round(rng.uniform(0.9, 1.0), 3),
                    diff_srv_rate = round(rng.uniform(0.0, 0.05), 3),
                    dst_host_count= rng.integers(200, 255),
                    dst_host_srv_count=rng.integers(200, 255),
                    dst_host_serror_rate=round(rng.uniform(0.8, 1.0), 3),
                    protocol_type = rng.choice(["tcp", "udp", "icmp"], p=[0.4, 0.2, 0.4]),
                    service       = rng.choice(["http", "private", "ecr_i", "smtp"]),
                    flag          = rng.choice(["SF", "S0", "RSTO"], p=[0.2, 0.6, 0.2]),
                    label         = label,
                )
            elif label == "probe":
                row = dict(
                    duration      = rng.integers(0, 20),
                    src_bytes     = rng.integers(0, 2000),
                    dst_bytes     = rng.integers(0, 2000),
                    land          = 0,
                    wrong_fragment= 0,
                    urgent        = 0,
                    hot           = rng.integers(0, 2),
                    num_failed_logins=0,
                    logged_in     = 0,
                    count         = rng.integers(100, 512),
                    srv_count     = rng.integers(1, 50),
                    serror_rate   = round(rng.uniform(0.0, 0.5), 3),
                    rerror_rate   = round(rng.uniform(0.0, 0.5), 3),
                    same_srv_rate = round(rng.uniform(0.0, 0.5), 3),
                    diff_srv_rate = round(rng.uniform(0.5, 1.0), 3),
                    dst_host_count= rng.integers(1, 100),
                    dst_host_srv_count=rng.integers(1, 50),
                    dst_host_serror_rate=round(rng.uniform(0.0, 0.3), 3),
                    protocol_type = rng.choice(["tcp", "udp", "icmp"], p=[0.5, 0.3, 0.2]),
                    service       = rng.choice(["http", "ftp", "private", "other"]),
                    flag          = rng.choice(["SF", "S0", "REJ", "RSTO"], p=[0.3, 0.3, 0.2, 0.2]),
                    label         = label,
                )
            elif label == "r2l":
                row = dict(
                    duration      = rng.integers(0, 600),
                    src_bytes     = rng.integers(100, 10000),
                    dst_bytes     = rng.integers(100, 10000),
                    land          = 0,
                    wrong_fragment= 0,
                    urgent        = 0,
                    hot           = rng.integers(0, 20),
                    num_failed_logins=rng.integers(1, 6),
                    logged_in     = rng.integers(0, 2),
                    count         = rng.integers(1, 20),
                    srv_count     = rng.integers(1, 20),
                    serror_rate   = round(rng.uniform(0.0, 0.2), 3),
                    rerror_rate   = round(rng.uniform(0.0, 0.2), 3),
                    same_srv_rate = round(rng.uniform(0.5, 1.0), 3),
                    diff_srv_rate = round(rng.uniform(0.0, 0.5), 3),
                    dst_host_count= rng.integers(1, 80),
                    dst_host_srv_count=rng.integers(1, 80),
                    dst_host_serror_rate=round(rng.uniform(0.0, 0.2), 3),
                    protocol_type = rng.choice(["tcp", "udp"], p=[0.8, 0.2]),
                    service       = rng.choice(["ftp", "smtp", "telnet", "ssh"]),
                    flag          = rng.choice(["SF", "REJ"], p=[0.7, 0.3]),
                    label         = label,
                )
            else:  # u2r
                row = dict(
                    duration      = rng.integers(0, 200),
                    src_bytes     = rng.integers(500, 20000),
                    dst_bytes     = rng.integers(500, 20000),
                    land          = 0,
                    wrong_fragment= 0,
                    urgent        = rng.integers(0, 4),
                    hot           = rng.integers(5, 30),
                    num_failed_logins=rng.integers(0, 3),
                    logged_in     = 1,
                    count         = rng.integers(1, 15),
                    srv_count     = rng.integers(1, 15),
                    serror_rate   = round(rng.uniform(0.0, 0.1), 3),
                    rerror_rate   = round(rng.uniform(0.0, 0.1), 3),
                    same_srv_rate = round(rng.uniform(0.6, 1.0), 3),
                    diff_srv_rate = round(rng.uniform(0.0, 0.4), 3),
                    dst_host_count= rng.integers(1, 50),
                    dst_host_srv_count=rng.integers(1, 50),
                    dst_host_serror_rate=round(rng.uniform(0.0, 0.1), 3),
                    protocol_type = "tcp",
                    service       = rng.choice(["telnet", "ssh", "ftp", "http"]),
                    flag          = "SF",
                    label         = label,
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """
    Encodes categorical features, splits data, trains three classifiers,
    and returns fitted models + evaluation metrics.
    No data leakage: LabelEncoders are fit ONLY on training split for categoricals.
    The scaler is fit inside the pipeline on training data only.
    """
    # --- Encode categoricals ---
    le_protocol = LabelEncoder()
    le_service  = LabelEncoder()
    le_flag     = LabelEncoder()
    le_label    = LabelEncoder()

    df = df.copy()
    df["protocol_type"] = le_protocol.fit_transform(df["protocol_type"])
    df["service"]       = le_service.fit_transform(df["service"])
    df["flag"]          = le_flag.fit_transform(df["flag"])
    df["label_enc"]     = le_label.fit_transform(df["label"])

    FEATURES = [
        "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "count",
        "srv_count", "serror_rate", "rerror_rate", "same_srv_rate",
        "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_serror_rate", "protocol_type", "service", "flag",
    ]

    X = df[FEATURES].values
    y = df["label_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    classifiers = {
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree":     DecisionTreeClassifier(max_depth=12, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    }

    results = {}
    for name, clf in classifiers.items():
        if name == "Logistic Regression":
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        else:
            pipe = Pipeline([("clf", clf)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results[name] = {
            "pipeline":   pipe,
            "accuracy":   round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision":  round(precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
            "recall":     round(recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
            "f1":         round(f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100, 2),
            "conf_matrix": confusion_matrix(y_test, y_pred),
            "report":     classification_report(y_test, y_pred, target_names=le_label.classes_, zero_division=0),
            "y_test":     y_test,
            "y_pred":     y_pred,
        }

    encoders = {
        "protocol": le_protocol,
        "service":  le_service,
        "flag":     le_flag,
        "label":    le_label,
        "features": FEATURES,
    }

    return results, encoders, X_test, y_test


# ─────────────────────────────────────────────
#  MATPLOTLIB DARK STYLE HELPER
# ─────────────────────────────────────────────
def dark_fig(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#cae8ff")
    return fig, ax


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ML Intrusion Detector")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Detect Traffic", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>⚠️ Demo mode — synthetic data.<br>"
        "Not a production IDS system.</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  LOAD DATA + TRAIN (cached — runs once)
# ─────────────────────────────────────────────
with st.spinner("🔄 Generating synthetic dataset and training models…"):
    df_raw    = generate_synthetic_data(n_samples=5000)
    results, encoders, X_test, y_test = train_models(df_raw)
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_pipeline   = results[best_model_name]["pipeline"]


# ════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("🛡️ ML Network Intrusion Detector")
    st.markdown(
        "A machine-learning dashboard that classifies network traffic as **normal** or a potential attack. "
        "Powered by Scikit-learn · Trained on **synthetic NSL-KDD-style data** (demo mode)."
    )
    st.info("ℹ️ **Demo mode:** All traffic records are synthetically generated for educational purposes. No real network packets are captured.")

    # ── Metric cards ──
    label_counts = df_raw["label"].value_counts()
    total        = len(df_raw)
    normal_cnt   = int(label_counts.get("normal", 0))
    attack_cnt   = total - normal_cnt
    dos_cnt      = int(label_counts.get("dos", 0))
    probe_cnt    = int(label_counts.get("probe", 0))
    r2l_cnt      = int(label_counts.get("r2l", 0))
    u2r_cnt      = int(label_counts.get("u2r", 0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Traffic Records", f"{total:,}")
    c2.metric("✅ Normal Traffic",         f"{normal_cnt:,}")
    c3.metric("🚨 Detected Attacks",       f"{attack_cnt:,}")
    c4.metric("🔎 Suspicious (Probe)",     f"{probe_cnt:,}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    # ── Donut / pie chart ──
    with col_left:
        st.subheader("Traffic Classification Distribution")
        labels_pie   = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        sizes        = [normal_cnt, dos_cnt, probe_cnt, r2l_cnt, u2r_cnt]
        colors_pie   = ["#3fb950", "#f85149", "#d29922", "#58a6ff", "#bc8cff"]
        fig, ax = dark_fig(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_pie, colors=colors_pie,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
            textprops={"color": "#c9d1d9"},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title("Traffic by Class", pad=14)
        st.pyplot(fig)
        plt.close(fig)

    # ── Attack breakdown bar chart ──
    with col_right:
        st.subheader("Attack Type Breakdown")
        attack_labels = ["DoS", "Probe", "R2L", "U2R"]
        attack_values = [dos_cnt, probe_cnt, r2l_cnt, u2r_cnt]
        bar_colors    = ["#f85149", "#d29922", "#58a6ff", "#bc8cff"]
        fig2, ax2 = dark_fig(figsize=(5, 4))
        bars = ax2.bar(attack_labels, attack_values, color=bar_colors, width=0.5, edgecolor="#0d1117")
        ax2.set_ylabel("Count")
        ax2.set_title("Attack Type Counts")
        ax2.grid(axis="y", color="#30363d", linestyle="--", linewidth=0.5)
        for bar, val in zip(bars, attack_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(val), ha="center", va="bottom", color="#c9d1d9", fontsize=9)
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Best model summary ──
    st.markdown("---")
    st.subheader(f"🏆 Best Model: {best_model_name}")
    m = results[best_model_name]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy",  f"{m['accuracy']}%")
    mc2.metric("Precision", f"{m['precision']}%")
    mc3.metric("Recall",    f"{m['recall']}%")
    mc4.metric("F1-Score",  f"{m['f1']}%")

    # ── Sample of dataset ──
    st.markdown("---")
    st.subheader("📋 Sample Traffic Records (first 10)")
    st.dataframe(
        df_raw.head(10).style.map(
            lambda v: "color: #3fb950" if v == "normal" else "color: #f85149",
            subset=["label"],
        ),
        use_container_width=True,
    )


# ════════════════════════════════════════════
#  PAGE: DETECT TRAFFIC
# ════════════════════════════════════════════
elif page == "🔍 Detect Traffic":

    st.title("🔍 Detect Network Traffic")
    st.markdown(
        "Enter the network connection features below and click **Analyze Traffic** to get a prediction. "
        "The model was trained on synthetic NSL-KDD-style data."
    )

    with st.form("predict_form"):
        st.subheader("Network Connection Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            duration          = st.number_input("Duration (seconds)", min_value=0, max_value=60000, value=0)
            src_bytes         = st.number_input("Source Bytes",        min_value=0, max_value=10_000_000, value=1000)
            dst_bytes         = st.number_input("Destination Bytes",   min_value=0, max_value=10_000_000, value=2000)
            protocol_type     = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
            service           = st.selectbox("Service", ["http", "ftp", "smtp", "ssh", "dns",
                                                         "telnet", "private", "ecr_i", "other"])

        with col2:
            flag              = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "SH", "OTH"])
            land              = st.selectbox("Land (same src/dst)", [0, 1])
            wrong_fragment    = st.number_input("Wrong Fragments", min_value=0, max_value=10, value=0)
            urgent            = st.number_input("Urgent Packets",  min_value=0, max_value=10, value=0)
            hot               = st.number_input("Hot Indicators",  min_value=0, max_value=100, value=0)

        with col3:
            num_failed_logins = st.number_input("Failed Logins",      min_value=0, max_value=10, value=0)
            logged_in         = st.selectbox("Logged In",             [0, 1])
            count             = st.number_input("Same-host count",    min_value=0, max_value=512, value=10)
            srv_count         = st.number_input("Same-service count", min_value=0, max_value=512, value=10)
            serror_rate       = st.slider("SYN Error Rate",   0.0, 1.0, 0.0, 0.01)

        col4, col5 = st.columns(2)
        with col4:
            rerror_rate       = st.slider("REJ Error Rate",      0.0, 1.0, 0.0, 0.01)
            same_srv_rate     = st.slider("Same Service Rate",   0.0, 1.0, 1.0, 0.01)
            diff_srv_rate     = st.slider("Diff Service Rate",   0.0, 1.0, 0.0, 0.01)
        with col5:
            dst_host_count    = st.number_input("Dst Host Count",     min_value=0, max_value=255, value=255)
            dst_host_srv_count= st.number_input("Dst Host Srv Count", min_value=0, max_value=255, value=255)
            dst_host_serror_rate = st.slider("Dst Host SYN Err Rate", 0.0, 1.0, 0.0, 0.01)

        submitted = st.form_submit_button("🚀 Analyze Traffic", use_container_width=True)

    if submitted:
        # --- Encode categoricals using fitted encoders ---
        def safe_encode(encoder, value):
            """Handle unseen labels by mapping to the first known class."""
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            return 0  # default to first class index

        proto_enc   = safe_encode(encoders["protocol"], protocol_type)
        service_enc = safe_encode(encoders["service"],  service)
        flag_enc    = safe_encode(encoders["flag"],     flag)

        input_vec = np.array([[
            duration, src_bytes, dst_bytes, land, wrong_fragment,
            urgent, hot, num_failed_logins, logged_in, count,
            srv_count, serror_rate, rerror_rate, same_srv_rate,
            diff_srv_rate, dst_host_count, dst_host_srv_count,
            dst_host_serror_rate, proto_enc, service_enc, flag_enc,
        ]])

        prediction_enc  = best_pipeline.predict(input_vec)[0]
        prediction_label = encoders["label"].inverse_transform([prediction_enc])[0]

        # Confidence (probability) — supported by RF and LR
        if hasattr(best_pipeline.named_steps.get("clf", best_pipeline[-1]), "predict_proba"):
            proba      = best_pipeline.predict_proba(input_vec)[0]
            confidence = round(float(proba.max()) * 100, 1)
        else:
            confidence = None

        st.markdown("---")
        st.subheader("🔎 Prediction Result")

        COLOR_MAP = {
            "normal": ("✅", "#3fb950", "NORMAL — Benign traffic"),
            "dos":    ("🚨", "#f85149", "ATTACK — Denial of Service"),
            "probe":  ("🔎", "#d29922", "SUSPICIOUS — Probe / Scan"),
            "r2l":    ("⚠️", "#58a6ff", "ATTACK — Remote to Local"),
            "u2r":    ("🔴", "#bc8cff", "ATTACK — User to Root"),
        }
        icon, color, verdict = COLOR_MAP.get(prediction_label, ("❓", "#8b949e", "UNKNOWN"))

        st.markdown(
            f"<div style='background:#161b22;border:2px solid {color};border-radius:12px;"
            f"padding:24px;text-align:center'>"
            f"<p style='font-size:2.5rem;margin:0'>{icon}</p>"
            f"<h2 style='color:{color};margin:8px 0'>{verdict}</h2>"
            f"<p style='color:#8b949e'>Raw class: <code>{prediction_label}</code>"
            + (f" &nbsp;|&nbsp; Confidence: <strong style='color:{color}'>{confidence}%</strong>" if confidence else "")
            + "</p></div>",
            unsafe_allow_html=True,
        )

        st.warning(
            "⚠️ **Disclaimer:** This prediction is an automated estimate from a model trained on **synthetic data**. "
            "It is NOT proof of an actual attack and should NOT be used for real network security decisions."
        )


# ════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════
elif page == "📈 Model Performance":

    st.title("📈 Model Performance")
    st.markdown(
        "Metrics are calculated on a **held-out 25% test set** not seen during training. "
        "Preprocessing (StandardScaler for Logistic Regression) is fit on training data only — no data leakage."
    )

    # ── Model comparison table ──
    st.subheader("Model Comparison")
    comp_data = {
        "Model":     list(results.keys()),
        "Accuracy":  [f"{v['accuracy']}%" for v in results.values()],
        "Precision": [f"{v['precision']}%" for v in results.values()],
        "Recall":    [f"{v['recall']}%" for v in results.values()],
        "F1-Score":  [f"{v['f1']}%" for v in results.values()],
    }
    st.dataframe(pd.DataFrame(comp_data).set_index("Model"), use_container_width=True)

    # ── Per-model deep dive ──
    st.markdown("---")
    selected_model = st.selectbox("Select model for detailed report:", list(results.keys()))
    r = results[selected_model]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{r['accuracy']}%")
    c2.metric("Precision", f"{r['precision']}%")
    c3.metric("Recall",    f"{r['recall']}%")
    c4.metric("F1-Score",  f"{r['f1']}%")

    st.markdown("---")
    col_cm, col_report = st.columns([1, 1])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = r["conf_matrix"]
        class_names = encoders["label"].classes_
        fig_cm, ax_cm = dark_fig(figsize=(5, 4))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks(range(len(class_names)))
        ax_cm.set_yticks(range(len(class_names)))
        ax_cm.set_xticklabels(class_names, rotation=30, ha="right", color="#c9d1d9", fontsize=9)
        ax_cm.set_yticklabels(class_names, color="#c9d1d9", fontsize=9)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Confusion Matrix — {selected_model}")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "#c9d1d9", fontsize=10)
        fig_cm.colorbar(im, ax=ax_cm)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_report:
        st.subheader("Classification Report")
        st.code(r["report"], language="text")

    # ── Feature importance (RF / DT only) ──
    clf_step = r["pipeline"].named_steps.get("clf", None)
    if clf_step and hasattr(clf_step, "feature_importances_"):
        st.markdown("---")
        st.subheader("Top 10 Feature Importances")
        importances = clf_step.feature_importances_
        indices     = np.argsort(importances)[::-1][:10]
        feat_names  = [encoders["features"][i] for i in indices]
        feat_vals   = importances[indices]

        fig_fi, ax_fi = dark_fig(figsize=(8, 3))
        ax_fi.barh(feat_names[::-1], feat_vals[::-1], color="#58a6ff", edgecolor="#0d1117")
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_title("Top 10 Features")
        ax_fi.grid(axis="x", color="#30363d", linestyle="--", linewidth=0.5)
        st.pyplot(fig_fi)
        plt.close(fig_fi)


# ════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    with st.expander("📖 What is Network Intrusion Detection?", expanded=True):
        st.markdown("""
**Network Intrusion Detection** is the process of monitoring network traffic for suspicious activity
and known threats. An **Intrusion Detection System (IDS)** alerts administrators when it finds
patterns that match known attacks or anomalies.

Traditional IDSes use rule-based signatures. **ML-based IDSes** learn patterns from labeled traffic data
and can detect novel threats without hand-crafted rules.
        """)

    with st.expander("🤖 Machine Learning Approach"):
        st.markdown("""
This project uses **supervised classification**:

| Step | Detail |
|------|--------|
| Dataset | Synthetic NSL-KDD-style data (5,000 records) |
| Features | 21 numeric + encoded categorical features |
| Algorithms | Random Forest · Decision Tree · Logistic Regression |
| Split | 75% train / 25% test (stratified) |
| Evaluation | Accuracy, Precision, Recall, F1-score, Confusion Matrix |

**Why Random Forest?**  
It handles class imbalance well, provides feature importances, and is robust to outliers.
        """)

    with st.expander("📂 Dataset Details"):
        st.markdown("""
**Source:** Inspired by the [NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html) (University of New Brunswick).

**Attack classes:**
- **Normal** — Legitimate network traffic
- **DoS** — Denial of Service (flood attacks that disrupt services)
- **Probe** — Port scanning and reconnaissance
- **R2L** — Remote-to-Local (unauthorized remote access attempts)
- **U2R** — User-to-Root (privilege escalation attempts)

> ⚠️ The data in this demo is **synthetically generated** to mimic NSL-KDD feature distributions.
> For real evaluations, download the official dataset from the UNB website.
        """)

    with st.expander("⚠️ Limitations"):
        st.markdown("""
1. **Demo data only** — Synthetic records do not represent real network traffic.
2. **No live capture** — This app does not sniff actual packets.
3. **Class imbalance** — R2L and U2R classes are rare; accuracy alone can be misleading.
4. **Dataset age** — NSL-KDD is based on 1999 traffic patterns; modern attacks differ significantly.
5. **Not production-ready** — Do not deploy this as a real IDS.
        """)

    with st.expander("🚀 Future Enhancements"):
        st.markdown("""
- Integrate the real NSL-KDD or CIC-IDS2017 dataset
- Add CSV batch prediction upload
- Store prediction history in SQLite
- Add SHAP explainability charts
- Deploy to Streamlit Cloud or Render
        """)

    st.markdown("---")
    st.markdown(
        "**Project:** BCA Data Science — ML Network Intrusion Detector  \n"
        "**Stack:** Python · Streamlit · Scikit-learn · Pandas · Matplotlib  \n"
        "**Purpose:** Academic demonstration only · Not a production security tool"
    )
