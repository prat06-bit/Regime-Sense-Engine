# Regime Sense Engine

Regime Sense Engine is a beginner-level machine learning project that identifies different market conditions (regimes) from price data.

Instead of assuming the market behaves the same way all the time, this project detects patterns such as stable periods, volatile phases, and trending movements using simple statistical features and unsupervised learning models.

The application includes an interactive Streamlit dashboard where users can upload data or generate synthetic market data to explore regime detection.

---

## Overview

This project:

* Reads price data (date and closing price)
* Calculates financial features like returns and volatility
* Uses machine learning to group data into different market regimes
* Displays results using charts and tables

---

## Models Used

### Hidden Markov Model (HMM)

* Detects hidden market states over time
* Shows probability of each regime
* Displays regime transitions

### KMeans Clustering

* Groups similar market conditions
* Serves as a simple baseline model
* Helps compare results with HMM

---

## Features

* Log Returns
* Rolling Volatility

These features help the model understand price movement and risk.

---

## Data Options

You can:

* Upload your own CSV file with `date` and `close` columns
* Generate synthetic market data inside the application





