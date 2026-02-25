Regime Sense Engine

Regime Sense Engine is a beginner-friendly machine learning project that helps identify different market conditions (regimes) from price data. Instead of assuming the market behaves the same way all the time, this project tries to detect patterns such as stable periods, volatile phases, or trending movements using simple statistical features and unsupervised learning models.

The application includes an interactive Streamlit dashboard where users can upload their own data or generate synthetic market data to explore how regime detection works.

 What This Project Does

Reads price data (date and closing price)

Calculates basic financial features like returns and volatility

Uses machine learning to group data into different market regimes

Displays results using simple charts and tables

This project is designed for learning purposes and for beginners who want to understand how AI can be applied to financial time series.

 Models Used
1️ Hidden Markov Model (HMM)

Finds hidden market states over time

Shows probabilities of each regime

Displays how regimes change from one state to another

2️ KMeans Clustering

Groups similar data points together

Easier to understand baseline method

Helps compare with HMM results

 Features Used

The model does not use raw prices directly. Instead, it calculates:

Log Returns – how much price changes from one day to the next

Rolling Volatility – how risky or unstable the market is over time

These features help the model understand market behavior.

 Data Options

You can use two types of data:

Upload your own CSV file with date and close columns

Generate synthetic data using a simple financial simulation
