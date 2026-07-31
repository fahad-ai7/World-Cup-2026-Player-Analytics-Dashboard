import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="World Cup 2026 Player Analytics", layout="wide"
)

st.title("⚽ World Cup 2026 Player Analytics Dashboard")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "wc2026_players_data.csv")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["goals_per_90"] = (df["goals"] / df["minutes_played"]) * 90
    df["assists_per_90"] = (df["assists"] / df["minutes_played"]) * 90
    df["goal_contributions_per_90"] = (
        (df["goals"] + df["assists"]) / df["minutes_played"]
    ) * 90
    return df


df_clean = load_data(file_path)

# Sidebar Filter
st.sidebar.header("Filter Players")
min_minutes = st.sidebar.slider(
    "Minimum Minutes Played", 0, int(df_clean["minutes_played"].max()), 180
)
filtered_df = df_clean[df_clean["minutes_played"] >= min_minutes]

# Calculate position averages on filtered data
pos_means = (
    filtered_df.groupby("position")[
        ["goals", "assists", "goal_contributions_per_90"]
    ]
    .mean()
    .round(2)
)

# ---------------------------------------------------------
# Display Charts in Grid Layout
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Avg Goal Contributions per 90")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(
        pos_means.index,
        pos_means["goal_contributions_per_90"],
        color="skyblue",
    )
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Per 90 Rate")
    st.pyplot(fig1)

with col2:
    st.subheader("Goal Contributions vs. Minutes Played")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(
        filtered_df["minutes_played"],
        filtered_df["goal_contributions_per_90"],
        alpha=0.6,
        c="teal",
    )
    ax2.axhline(
        filtered_df["goal_contributions_per_90"].mean(),
        color="red",
        linestyle="--",
    )
    ax2.set_xlabel("Minutes Played")
    ax2.set_ylabel("Per 90 Rate")
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Age Distribution")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(
        filtered_df["age"], bins=15, color="mediumseagreen", edgecolor="black"
    )
    ax3.set_xlabel("Age")
    st.pyplot(fig3)

with col4:
    st.subheader("Pass Accuracy by Position")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    positions = filtered_df["position"].dropna().unique()
    data_by_pos = [
        filtered_df[filtered_df["position"] == pos]["pass_accuracy_pct"].dropna()
        for pos in positions
    ]
    ax4.boxplot(data_by_pos, tick_labels=positions)
    ax4.set_ylabel("Pass Accuracy (%)")
    st.pyplot(fig4)

# Data Table
st.subheader("Top Performers Data View")
st.dataframe(
    filtered_df[
        [
            "name",
            "position",
            "club",
            "minutes_played",
            "goals",
            "assists",
            "goal_contributions_per_90",
        ]
    ].sort_values("goal_contributions_per_90", ascending=False)
)