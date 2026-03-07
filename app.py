import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
from io import StringIO
import datetime


def generate_gbm(start_price=100, mu=0.08, sigma=0.2, days=500, seed=42):
    np.random.seed(seed)
    dt = 1/252
    prices = [start_price]
    for _ in range(days-1):
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal()
        price = prices[-1] * np.exp(drift + shock)
        prices.append(price)
    dates = pd.date_range(end=datetime.date.today(), periods=days)
    return pd.DataFrame({'date': dates, 'close': prices})


def compute_features(df, window=20):
    df = df.copy()
    df['log_return'] = np.log(df['close']).diff()
    df['volatility'] = df['log_return'].rolling(window).std() * np.sqrt(252)
    df = df.dropna().reset_index(drop=True)
    features = df[['log_return', 'volatility']].values
    return df, features

# HMM Model
def fit_hmm(features, n_states=3, random_state=42):
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=random_state)
    model.fit(features)
    hidden_states = model.predict(features)
    regime_probs = model.predict_proba(features)
    transmat = model.transmat_
    means = model.means_
    covars = np.sqrt(np.array([np.diag(c) for c in model.covars_]))
    return hidden_states, regime_probs, transmat, means, covars

# KMeans Model
def fit_kmeans(features, n_states=3, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_
    return labels, centers

# Regime Statistics
def compute_regime_stats(df, labels, n_states):
    stats = []
    for i in range(n_states):
        mask = labels == i
        mean_ret = df.loc[mask, 'log_return'].mean()
        vol = df.loc[mask, 'log_return'].std() * np.sqrt(252)
        stats.append({'Regime': i, 'Mean Return': mean_ret, 'Volatility': vol, 'Count': mask.sum()})
    return pd.DataFrame(stats)


def plot_regimes(df, labels, title="Price Chart by Regime"):
    plt.figure(figsize=(14,6))
    palette = sns.color_palette("Set1", np.unique(labels).max()+1)
    for regime in np.unique(labels):
        mask = labels == regime
        plt.plot(df['date'][mask], df['close'][mask], '.', label=f'Regime {regime}', color=palette[regime], alpha=0.7)
    plt.plot(df['date'], df['close'], color='gray', alpha=0.3, linewidth=1)
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

def plot_regime_probs(df, regime_probs):
    plt.figure(figsize=(14,4))
    for i in range(regime_probs.shape[1]):
        plt.plot(df['date'], regime_probs[:,i], label=f'Regime {i}')
    plt.title('Regime Probabilities')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_transition_matrix(transmat):
    plt.figure(figsize=(6,5))
    sns.heatmap(transmat, annot=True, cmap='Blues', fmt='.2f', square=True)
    plt.title('Regime Transition Matrix')
    plt.xlabel('To Regime')
    plt.ylabel('From Regime')
    plt.tight_layout()
    plt.show()

# Streamlit 
def run_streamlit():
    st.set_page_config(layout="wide", page_title="Market Regime Detection Engine")
    st.title("Market Regime Detection Engine")
    st.markdown("Academic-style regime detection using HMM and KMeans.")

    with st.sidebar:
        st.header("Configuration")
        n_states = st.slider("Number of Regimes", 2, 6, 3)
        window = st.slider("Rolling Window Size", 5, 60, 20)
        model_type = st.selectbox("Model", ["HMM", "KMeans"])
        st.markdown("---")
        st.subheader("Data Source")
        data_mode = st.radio("Choose Data", ["Upload CSV", "Synthetic GBM"])
        if data_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV (date, close)", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file, parse_dates=['date'])
        else:
            st.write("Synthetic GBM Parameters:")
            start_price = st.number_input("Start Price", 1, 10000, 100)
            mu = st.number_input("Drift (mu)", 0.0, 1.0, 0.08, step=0.01)
            sigma = st.number_input("Volatility (sigma)", 0.01, 1.0, 0.2, step=0.01)
            days = st.number_input("Days", 50, 2000, 500)
            seed = st.number_input("Random Seed", 0, 9999, 42)
            df = generate_gbm(start_price, mu, sigma, days, seed)

    if 'df' not in locals():
        st.warning("Please upload a CSV or generate synthetic data.")
        st.stop()

    df, features = compute_features(df, window)

    if model_type == "HMM":
        labels, regime_probs, transmat, means, covars = fit_hmm(features, n_states)
        st.subheader("HMM Regime Detection")
        st.write(f"Transition Matrix:")
        st.dataframe(pd.DataFrame(transmat, columns=[f"To {i}" for i in range(n_states)], index=[f"From {i}" for i in range(n_states)]))
        st.write("State Means:")
        st.dataframe(pd.DataFrame(means, columns=['Mean Return', 'Volatility']))
        st.write("State Volatilities:")
        st.dataframe(pd.DataFrame(covars, columns=['Return Std', 'Volatility Std']))
    else:
        labels, centers = fit_kmeans(features, n_states)
        regime_probs = np.zeros((len(labels), n_states))
        regime_probs[np.arange(len(labels)), labels] = 1
        transmat = None
        st.subheader("KMeans Regime Detection")
        st.write("Cluster Centers:")
        st.dataframe(pd.DataFrame(centers, columns=['Mean Return', 'Volatility']))

    stats = compute_regime_stats(df, labels, n_states)
    st.write("Regime Statistics:")
    st.dataframe(stats)

    # Plots
    st.subheader("Visualizations")
    import plotly.express as px
    import plotly.graph_objects as go
    # Price chart colored by regime
    fig = px.scatter(df, x='date', y='close', color=labels.astype(str), title="Price Chart by Regime", color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    for i in range(regime_probs.shape[1]):
        fig2.add_trace(go.Scatter(x=df['date'], y=regime_probs[:,i], mode='lines', name=f'Regime {i}'))
    fig2.update_layout(title="Regime Probabilities", xaxis_title="Date", yaxis_title="Probability")
    st.plotly_chart(fig2, use_container_width=True)

    if transmat is not None:
        fig3 = px.imshow(transmat, text_auto='.2f', color_continuous_scale='Blues', title="Transition Matrix")
        st.plotly_chart(fig3, use_container_width=True)

def main():
    import streamlit.runtime.scriptrunner as st_runtime
    if st_runtime.get_script_run_ctx() is not None:
        run_streamlit()
        return
    import argparse
    parser = argparse.ArgumentParser(description="Market Regime Detection Engine")
    parser.add_argument('--csv', type=str, help='Path to CSV file (date, close)')
    parser.add_argument('--gbm', action='store_true', help='Use synthetic GBM data')
    parser.add_argument('--n_states', type=int, default=3, help='Number of regimes')
    parser.add_argument('--window', type=int, default=20, help='Rolling window size')
    parser.add_argument('--model', type=str, default='HMM', choices=['HMM', 'KMeans'], help='Model type')
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=['date'])
    elif args.gbm:
        df = generate_gbm()
    else:
        print("Please provide --csv or --gbm")
        return

    df, features = compute_features(df, args.window)

    if args.model == 'HMM':
        labels, regime_probs, transmat, means, covars = fit_hmm(features, args.n_states)
        print("Transition Matrix:\n", transmat)
        print("State Means:\n", means)
        print("State Volatilities:\n", covars)
    else:
        labels, centers = fit_kmeans(features, args.n_states)
        regime_probs = np.zeros((len(labels), args.n_states))
        regime_probs[np.arange(len(labels)), labels] = 1
        transmat = None
        print("Cluster Centers:\n", centers)

    stats = compute_regime_stats(df, labels, args.n_states)
    print("Regime Statistics:\n", stats)
    
    plot_regimes(df, labels)
    plot_regime_probs(df, regime_probs)
    if transmat is not None:
        plot_transition_matrix(transmat)

if __name__ == "__main__":
    main()
