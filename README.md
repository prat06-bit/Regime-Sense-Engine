Regime Sense Engine

Regime Sense Engine is a beginner-friendly machine learning project that helps identify different market conditions (regimes) from price data.

Instead of assuming the market behaves the same way all the time, this project detects patterns such as stable periods, volatile phases, and trending movements using simple statistical features and unsupervised learning models.

The application includes an interactive Streamlit dashboard where users can upload data or generate synthetic market data to explore how regime detection works.

What This Project Does

Reads price data (date and closing price)

Calculates basic financial features like returns and volatility

Uses machine learning to group data into different market regimes

Displays results using simple charts and tables

This project is designed for learning purposes and for beginners who want to understand how AI can be applied to financial time series.

Models Used
🔹 Hidden Markov Model (HMM)

Finds hidden market states over time

Shows probabilities of each regime

Displays how regimes change from one state to another

🔹 KMeans Clustering

Groups similar data points together

Beginner-friendly baseline model

Helps compare results with HMM

Features Used

The model uses simple statistical features:

Log Returns — daily price changes

Rolling Volatility — how risky the market is over time
 Data Options

You can choose between:

Uploading your own CSV file (date, close)

Generating synthetic market data inside the app
