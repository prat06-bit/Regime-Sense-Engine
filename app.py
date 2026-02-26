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

def run_streamlit() -> None:
    st.set_page_config(layout="wide", page_title="Market Regime Detection Engine")
    st.title("Market Regime Detection Engine")
    st.caption("Regime detection using Hidden Markov Models and K-Means clustering.")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        n_states = st.slider("Number of Regimes", 2, 6, 3)
        window = st.slider("Rolling Window Size", 5, 60, 20)
        model_type = st.selectbox("Model", ["HMM", "KMeans"])

        st.markdown("---")
        st.subheader("Data Source")
        data_mode = st.radio("Choose Data", ["Synthetic GBM", "Upload CSV"])

    # ── Data loading ─────────────────────────────────────────────────────────
    df: Optional[pd.DataFrame] = None

    if data_mode == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload CSV (date, close)", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded, parse_dates=["date"])
                if not {"date", "close"}.issubset(df.columns):
                    st.error("CSV must contain 'date' and 'close' columns.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
                st.stop()
    else:
        with st.sidebar:
            st.write("GBM Parameters:")
            start_price = st.number_input("Start Price", 1, 10_000, 100)
            mu = st.number_input("Drift (mu)", 0.0, 1.0, 0.08, step=0.01)
            sigma = st.number_input("Volatility (sigma)", 0.01, 1.0, 0.2, step=0.01)
            days = st.number_input("Days", 50, 2_000, 500)
            seed = st.number_input("Random Seed", 0, 9_999, 42)
        df = generate_gbm(start_price, mu, sigma, int(days), int(seed))

    if df is None:
        st.info("Upload a CSV or configure synthetic GBM data in the sidebar.")
        st.stop()

    df, features = compute_features(df, window)

    # ── Model fitting ─────────────────────────────────────────────────────────
    transmat: Optional[np.ndarray] = None

    if model_type == "HMM":
        labels, regime_probs, transmat, means, covars = fit_hmm(features, n_states)
        with st.expander("HMM Model Parameters", expanded=False):
            col1, col2 = st.columns(2)
            col1.write("**State Means**")
            col1.dataframe(pd.DataFrame(means, columns=["Mean Return", "Volatility"]))
            col2.write("**State Std Devs**")
            col2.dataframe(pd.DataFrame(covars, columns=["Return Std", "Vol Std"]))
            st.write("**Transition Matrix**")
            st.dataframe(
                pd.DataFrame(
                    transmat,
                    columns=[f"To {i}" for i in range(n_states)],
                    index=[f"From {i}" for i in range(n_states)],
                ).style.format("{:.3f}")
            )
    else:
        labels, centers = fit_kmeans(features, n_states)
        regime_probs = np.eye(n_states)[labels]
        with st.expander("KMeans Cluster Centers", expanded=False):
            st.dataframe(pd.DataFrame(centers, columns=["Mean Return", "Volatility"]))

    # ── Regime stats ──────────────────────────────────────────────────────────
    st.subheader("Regime Statistics")
    st.dataframe(
        compute_regime_stats(df, labels, n_states).style.format(
            {"Mean Return": "{:.4f}", "Volatility (Ann.)": "{:.4f}", "Weight (%)": "{:.1f}"}
        ),
        use_container_width=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("Visualizations")
    st.plotly_chart(chart_price(df, labels), use_container_width=True)
    st.plotly_chart(chart_regime_probs(df, regime_probs), use_container_width=True)
    if transmat is not None:
        st.plotly_chart(chart_transition_matrix(transmat), use_container_width=True)


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