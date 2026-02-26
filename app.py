import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── Data Generation ───────────────────────────────────────────────────────────

@st.cache_data
def generate_gbm(
    start_price: float = 100,
    mu: float = 0.08,
    sigma: float = 0.2,
    days: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Vectorized Geometric Brownian Motion price simulation."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal(0, 1, days - 1)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    prices = np.empty(days)
    prices[0] = start_price
    np.exp(log_returns, out=prices[1:])
    np.cumprod(prices, out=prices)               # avoids Python loop entirely
    prices[1:] *= start_price / prices[0]        # re-anchor after cumprod
    # Simpler: use cumsum on log-returns then exponentiate
    prices = start_price * np.exp(np.concatenate([[0], np.cumsum(log_returns)]))
    dates = pd.date_range(end=datetime.date.today(), periods=days)
    return pd.DataFrame({"date": dates, "close": prices})


# ── Feature Engineering ───────────────────────────────────────────────────────

@st.cache_data
def compute_features(df: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    df["log_return"] = np.log(df["close"]).diff()
    df["volatility"] = df["log_return"].rolling(window).std() * np.sqrt(252)
    df = df.dropna().reset_index(drop=True)
    return df, df[["log_return", "volatility"]].values


# ── Models ────────────────────────────────────────────────────────────────────

@st.cache_resource
def fit_hmm(
    features: np.ndarray, n_states: int = 3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=random_state,
    )
    model.fit(features)
    hidden_states = model.predict(features)
    regime_probs = model.predict_proba(features)
    covars = np.sqrt(np.array([np.diag(c) for c in model.covars_]))
    return hidden_states, regime_probs, model.transmat_, model.means_, covars


@st.cache_resource
def fit_kmeans(
    features: np.ndarray, n_states: int = 3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    X_scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans.cluster_centers_


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_regime_stats(df: pd.DataFrame, labels: np.ndarray, n_states: int) -> pd.DataFrame:
    rows = []
    for i in range(n_states):
        mask = labels == i
        rows.append({
            "Regime": i,
            "Mean Return": df.loc[mask, "log_return"].mean(),
            "Volatility (Ann.)": df.loc[mask, "log_return"].std() * np.sqrt(252),
            "Count": int(mask.sum()),
            "Weight (%)": round(100 * mask.mean(), 1),
        })
    return pd.DataFrame(rows)


# ── Visualizations ────────────────────────────────────────────────────────────

COLORS = px.colors.qualitative.Set1


def _regime_color_map(n_states: int) -> dict:
    return {str(i): COLORS[i % len(COLORS)] for i in range(n_states)}


def chart_price(df: pd.DataFrame, labels: np.ndarray) -> go.Figure:
    return px.scatter(
        df,
        x="date",
        y="close",
        color=labels.astype(str),
        title="Price Chart by Regime",
        color_discrete_map=_regime_color_map(labels.max() + 1),
        labels={"color": "Regime"},
    )


def chart_regime_probs(df: pd.DataFrame, regime_probs: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for i in range(regime_probs.shape[1]):
        fig.add_trace(
            go.Scatter(x=df["date"], y=regime_probs[:, i], mode="lines", name=f"Regime {i}")
        )
    fig.update_layout(
        title="Regime Probabilities", xaxis_title="Date", yaxis_title="Probability"
    )
    return fig


def chart_transition_matrix(transmat: np.ndarray) -> go.Figure:
    n = transmat.shape[0]
    return px.imshow(
        transmat,
        text_auto=".2f",
        color_continuous_scale="Blues",
        title="Regime Transition Matrix",
        labels={"x": "To Regime", "y": "From Regime"},
        x=[f"Regime {i}" for i in range(n)],
        y=[f"Regime {i}" for i in range(n)],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Market Regime Detection Engine")
    parser.add_argument("--csv", type=str, help="Path to CSV file (date, close)")
    parser.add_argument("--gbm", action="store_true", help="Use synthetic GBM data")
    parser.add_argument("--n_states", type=int, default=3)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--model", type=str, default="HMM", choices=["HMM", "KMeans"])
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"])
    elif args.gbm:
        df = generate_gbm()
    else:
        print("Provide --csv <path> or --gbm")
        return

    df, features = compute_features(df, args.window)
    transmat: Optional[np.ndarray] = None

    if args.model == "HMM":
        labels, regime_probs, transmat, means, covars = fit_hmm(features, args.n_states)
        print("Transition Matrix:\n", transmat)
        print("State Means:\n", means)
        print("State Std Devs:\n", covars)
    else:
        labels, centers = fit_kmeans(features, args.n_states)
        regime_probs = np.eye(args.n_states)[labels]
        print("Cluster Centers:\n", centers)

    print("\nRegime Statistics:\n", compute_regime_stats(df, labels, args.n_states))


# ── Streamlit UI ──────────────────────────────────────────────────────────────

HERO_CSS = """
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f2027, #1a3a4a, #0f2027);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.8rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p { color: #94a3b8; font-size: 1.05rem; line-height: 1.7; margin: 0; }

/* ── Feature cards ── */
.cards { display: flex; gap: 1rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.card {
    flex: 1; min-width: 200px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.card-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.card-title { color: #e2e8f0; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; }
.card-desc { color: #64748b; font-size: 0.83rem; line-height: 1.5; }

/* ── Upload zone ── */
.upload-box {
    background: #0d1117;
    border: 2px dashed #38bdf8;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.upload-box h3 { color: #38bdf8; margin-bottom: 0.3rem; }
.upload-box p  { color: #64748b; font-size: 0.88rem; margin: 0; }

/* ── Section headings ── */
.section-label {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.6rem;
    border-left: 3px solid #38bdf8;
    padding-left: 0.6rem;
}

/* ── Metric pills ── */
.pill {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-right: 0.4rem;
}
</style>
"""

def run_streamlit() -> None:
    st.set_page_config(
        layout="wide",
        page_title="Market Regime Detection Engine",
        page_icon="📈",
    )
    st.markdown(HERO_CSS, unsafe_allow_html=True)

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>📈 Market Regime Detection Engine</h1>
        <p>
            Financial markets cycle through distinct behavioural states — trending bull runs,
            high-volatility crashes, and quiet consolidation phases. This engine automatically
            identifies those hidden regimes in any price series using two complementary
            machine-learning models, giving you a quantitative edge in risk management,
            strategy switching, and portfolio allocation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cards">
        <div class="card">
            <div class="card-icon">🔮</div>
            <div class="card-title">Hidden Markov Model</div>
            <div class="card-desc">Probabilistic transitions between latent market states — captures regime stickiness and provides soft probabilities per day.</div>
        </div>
        <div class="card">
            <div class="card-icon">🎯</div>
            <div class="card-title">K-Means Clustering</div>
            <div class="card-desc">Fast unsupervised clustering on return & volatility features — great for exploratory segmentation with no distributional assumptions.</div>
        </div>
        <div class="card">
            <div class="card-icon">📊</div>
            <div class="card-title">Regime Analytics</div>
            <div class="card-desc">Per-regime return, annualised volatility, weight, and transition probabilities surfaced in interactive Plotly charts.</div>
        </div>
        <div class="card">
            <div class="card-icon">🧪</div>
            <div class="card-title">Synthetic GBM</div>
            <div class="card-desc">No data? Generate realistic Geometric Brownian Motion price paths to explore the engine before plugging in real tickers.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: model config ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        model_type = st.selectbox("Detection Model", ["HMM", "KMeans"])
        n_states = st.slider("Number of Regimes", 2, 6, 3)
        window = st.slider("Rolling Window (days)", 5, 60, 20)
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        data_mode = st.radio("Choose Data", ["Synthetic GBM", "Upload CSV"])

    # ── Data loading ─────────────────────────────────────────────────────────
    df: Optional[pd.DataFrame] = None

    if data_mode == "Upload CSV":
        # ── Inline upload zone on main canvas ────────────────────────────────
        st.markdown('<div class="section-label">Load your data</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="upload-box">
            <h3>⬆️ Upload a Price CSV</h3>
            <p>File must contain two columns: <strong>date</strong> (YYYY-MM-DD) and <strong>close</strong> (numeric price)</p>
        </div>
        """, unsafe_allow_html=True)

        col_up, col_pad = st.columns([1, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Drop your CSV here or click to browse",
                type=["csv"],
                label_visibility="collapsed",
            )

        # Also keep sidebar uploader as a convenience
        sidebar_uploaded = st.sidebar.file_uploader("Or upload from sidebar", type=["csv"])
        uploaded = uploaded or sidebar_uploaded

        if uploaded:
            try:
                df = pd.read_csv(uploaded, parse_dates=["date"])
                if not {"date", "close"}.issubset(df.columns):
                    st.error("❌ CSV must contain **date** and **close** columns.")
                    st.stop()
                st.success(f"✅ Loaded **{len(df):,}** rows from `{uploaded.name}`")
            except Exception as e:
                st.error(f"❌ Failed to parse CSV: {e}")
                st.stop()
        else:
            st.info("👆 Upload a CSV above to get started, or switch to **Synthetic GBM** in the sidebar.")
            st.stop()

    else:  # Synthetic GBM
        st.markdown('<div class="section-label">Synthetic GBM parameters</div>', unsafe_allow_html=True)
        with st.sidebar:
            start_price = st.number_input("Start Price ($)", 1, 10_000, 100)
            mu    = st.number_input("Annual Drift (μ)", 0.0, 1.0, 0.08, step=0.01)
            sigma = st.number_input("Annual Volatility (σ)", 0.01, 1.0, 0.2, step=0.01)
            days  = st.number_input("Trading Days", 50, 2_000, 500)
            seed  = st.number_input("Random Seed", 0, 9_999, 42)
        df = generate_gbm(start_price, mu, sigma, int(days), int(seed))
        st.markdown(
            f'<span class="pill">🏦 Start price ${start_price}</span>'
            f'<span class="pill">📈 μ = {mu:.0%}</span>'
            f'<span class="pill">〰️ σ = {sigma:.0%}</span>'
            f'<span class="pill">📅 {int(days)} days</span>',
            unsafe_allow_html=True,
        )

    df, features = compute_features(df, window)

    # ── Model fitting ─────────────────────────────────────────────────────────
    transmat: Optional[np.ndarray] = None
    st.markdown('<div class="section-label">Model output</div>', unsafe_allow_html=True)

    if model_type == "HMM":
        labels, regime_probs, transmat, means, covars = fit_hmm(features, n_states)
        with st.expander("🔮 HMM Model Parameters", expanded=False):
            col1, col2 = st.columns(2)
            col1.markdown("**State Means**")
            col1.dataframe(pd.DataFrame(means, columns=["Mean Return", "Volatility"]))
            col2.markdown("**State Std Devs**")
            col2.dataframe(pd.DataFrame(covars, columns=["Return Std", "Vol Std"]))
            st.markdown("**Transition Matrix**")
            st.dataframe(
                pd.DataFrame(
                    transmat,
                    columns=[f"→ Regime {i}" for i in range(n_states)],
                    index=[f"Regime {i}" for i in range(n_states)],
                ).style.format("{:.3f}").background_gradient(cmap="Blues")
            )
    else:
        labels, centers = fit_kmeans(features, n_states)
        regime_probs = np.eye(n_states)[labels]
        with st.expander("🎯 KMeans Cluster Centers", expanded=False):
            st.dataframe(pd.DataFrame(centers, columns=["Mean Return", "Volatility"]))

    # ── Regime stats ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Regime statistics</div>', unsafe_allow_html=True)
    stats = compute_regime_stats(df, labels, n_states)
    st.dataframe(
        stats.style
            .format({"Mean Return": "{:.4f}", "Volatility (Ann.)": "{:.4f}", "Weight (%)": "{:.1f}"})
            .background_gradient(subset=["Volatility (Ann.)"], cmap="RdYlGn_r")
            .background_gradient(subset=["Mean Return"], cmap="RdYlGn"),
        use_container_width=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Visualizations</div>', unsafe_allow_html=True)

    chart_cfg = dict(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#94a3b8",
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
    )

    fig_price = chart_price(df, labels)
    fig_price.update_layout(**chart_cfg)
    st.plotly_chart(fig_price, use_container_width=True)

    fig_probs = chart_regime_probs(df, regime_probs)
    fig_probs.update_layout(**chart_cfg)
    st.plotly_chart(fig_probs, use_container_width=True)

    if transmat is not None:
        fig_tm = chart_transition_matrix(transmat)
        fig_tm.update_layout(paper_bgcolor="#0d1117", font_color="#94a3b8")
        st.plotly_chart(fig_tm, use_container_width=True)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import streamlit.runtime.scriptrunner as st_runtime
        if st_runtime.get_script_run_ctx() is not None:
            run_streamlit()
            return
    except ImportError:
        pass
    run_cli()


if __name__ == "__main__":
    main()
