import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Environment & Setup
# ---------------------------------------------------------
print("--- Environment Versions ---")
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)

# ---------------------------------------------------------
# 2. Data Loading & Inspection
# ---------------------------------------------------------
# Get directory where players.py lives to make paths bulletproof
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n--- Data Directory Contents ---")
if os.path.exists(BASE_DIR):
    print(os.listdir(BASE_DIR))
else:
    print(f"Directory '{BASE_DIR}' not found.")

# Construct paths relative to players.py location
file1_path = os.path.join(BASE_DIR, "wc2026_players_1_data.csv")
file2_path = os.path.join(BASE_DIR, "wc2026_players_data.csv")

# Load datasets
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

print("\n--- Dataset Dimensions ---")
print("File 1 Shape:", df1.shape)
print("File 2 Shape:", df2.shape)

print("\n--- Columns ---")
print("FILE 1 COLUMNS:", df1.columns.tolist())
print("FILE 2 COLUMNS:", df2.columns.tolist())

print("\n--- Head Preview (File 1) ---")
print(df1.head(10))

print("\n--- Head Preview (File 2) ---")
print(df2.head(10))

# ---------------------------------------------------------
# 3. Data Integrity & Overlap Check
# ---------------------------------------------------------
print("\n--- Overlap & Duplicate Checks ---")
common_ids = set(df1["player_id"]) & set(df2["player_id"])
print("Players in File 1:", len(df1))
print("Players in File 2:", len(df2))
print("Players appearing in both:", len(common_ids))

# Primary analysis dataset (df2)
df = pd.read_csv(file2_path)

print("\n--- Primary Dataset Info ---")
print("Dataset shape:", df.shape)
df.info()

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate player IDs:", df["player_id"].duplicated().sum())

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Categorical Summaries ---")
print("Positions:\n", df["position"].value_counts())
print("Unique Countries:", df["country"].nunique())
print("Unique Clubs:", df["club"].nunique())

print("\n--- Metric Ranges ---")
print("Age range:", df["age"].min(), "to", df["age"].max())
print(
    "Matches range:",
    df["matches_played"].min(),
    "to",
    df["matches_played"].max(),
)
print(
    "Minutes range:",
    df["minutes_played"].min(),
    "to",
    df["minutes_played"].max(),
)
print(
    "Pass accuracy range:",
    df["pass_accuracy_pct"].min(),
    "to",
    df["pass_accuracy_pct"].max(),
)
print("Goals range:", df["goals"].min(), "to", df["goals"].max())
print("Assists range:", df["assists"].min(), "to", df["assists"].max())

# ---------------------------------------------------------
# 4. Data Cleaning & Feature Engineering
# ---------------------------------------------------------
df_clean = df.copy()

print("\n--- Data Cleaning Verification ---")
print("Original shape:", df.shape)
print("Clean dataset shape:", df_clean.shape)

# Verify no leading/trailing whitespace in string columns
print(df_clean["name"].str.strip().equals(df_clean["name"]))
print(df_clean["country"].str.strip().equals(df_clean["country"]))
print(df_clean["position"].str.strip().equals(df_clean["position"]))
print(df_clean["club"].str.strip().equals(df_clean["club"]))

# Calculate normalized per-90 metrics
df_clean["goals_per_90"] = (
    df_clean["goals"] / df_clean["minutes_played"]
) * 90
df_clean["assists_per_90"] = (
    df_clean["assists"] / df_clean["minutes_played"]
) * 90
df_clean["goal_contributions_per_90"] = (
    (df_clean["goals"] + df_clean["assists"]) / df_clean["minutes_played"]
) * 90
df_clean["shots_on_target_per_90"] = (
    df_clean["shots_on_target"] / df_clean["minutes_played"]
) * 90

# Sub-dataframe for per-90 metrics to exclude tiny sample sizes (minimum 180 mins)
df_qualified = df_clean[df_clean["minutes_played"] >= 180].copy()

# ---------------------------------------------------------
# 5. Data Analysis & Insights
# ---------------------------------------------------------
print("\n--- Top 10 Goals per 90 (Min 180 Mins Played) ---")
print(
    df_qualified[["name", "goals", "minutes_played", "goals_per_90"]].head(10)
)

print("\n--- Top 10 Assists per 90 (Min 180 Mins Played) ---")
print(
    df_qualified[["name", "assists", "minutes_played", "assists_per_90"]].head(
        10
    )
)

print(
    "\n--- Top 10 Goal Contributions per 90 (Min 180 Mins Played) ---"
)
print(
    df_qualified[
        [
            "name",
            "goals",
            "assists",
            "minutes_played",
            "goal_contributions_per_90",
        ]
    ].head(10)
)

print("\n--- Top 10 Shots on Target per 90 (Min 180 Mins Played) ---")
print(
    df_qualified[
        ["name", "shots_on_target", "minutes_played", "shots_on_target_per_90"]
    ].head(10)
)

print("\n--- Highest Goal Contributions per 90 Overall ---")
print(
    df_qualified[["name", "goal_contributions_per_90"]]
    .sort_values("goal_contributions_per_90", ascending=False)
    .head(10)
)

print("\n--- Top Players by Minutes Played ---")
print(
    df_clean[
        [
            "name",
            "minutes_played",
            "goals",
            "assists",
            "goal_contributions_per_90",
        ]
    ]
    .sort_values("minutes_played", ascending=False)
    .head(10)
)

print("\n--- Top 15 Goal Contributors per 90 (with position) ---")
print(
    df_qualified[
        ["name", "position", "goals", "assists", "goal_contributions_per_90"]
    ]
    .sort_values("goal_contributions_per_90", ascending=False)
    .head(15)
)

print("\n--- Position Averages ---")
pos_means = (
    df_clean.groupby("position")[
        ["goals", "assists", "goal_contributions_per_90"]
    ]
    .mean()
    .round(2)
)
print(pos_means)

# ---------------------------------------------------------
# 6. Data Visualizations (Separate Figures & Auto-Save)
# ---------------------------------------------------------

# Chart 1: Average Goal Contributions per 90 by Position
plt.figure(figsize=(8, 5))
plt.bar(
    pos_means.index, pos_means["goal_contributions_per_90"], color="skyblue"
)
plt.title("Average Goal Contributions per 90 by Position")
plt.xlabel("Position")
plt.ylabel("Goal Contributions per 90")
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "1_avg_goal_contributions.png"), dpi=300
)
plt.show()

# Chart 2: Goal Contributions vs. Minutes Played
plt.figure(figsize=(9, 5))
plt.scatter(
    df_clean["minutes_played"],
    df_clean["goal_contributions_per_90"],
    alpha=0.6,
    c="teal",
    edgecolor="k",
)
plt.axhline(
    df_clean["goal_contributions_per_90"].mean(),
    color="red",
    linestyle="--",
    label="Average",
)
plt.title("Goal Contributions per 90 vs. Minutes Played")
plt.xlabel("Minutes Played")
plt.ylabel("Goal Contributions per 90")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "2_contributions_vs_minutes.png"), dpi=300
)
plt.show()

# Chart 3: Distribution of Player Ages
plt.figure(figsize=(8, 5))
plt.hist(df_clean["age"], bins=15, color="mediumseagreen", edgecolor="black")
plt.title("Distribution of Player Ages")
plt.xlabel("Age")
plt.ylabel("Number of Players")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "3_age_distribution.png"), dpi=300)
plt.show()

# Chart 4: Pass Accuracy Distribution Across Positions
plt.figure(figsize=(9, 5))
positions = df_clean["position"].dropna().unique()
data_by_pos = [
    df_clean[df_clean["position"] == pos]["pass_accuracy_pct"].dropna()
    for pos in positions
]
plt.boxplot(data_by_pos, tick_labels=positions)
plt.title("Pass Accuracy Distribution by Position")
plt.xlabel("Position")
plt.ylabel("Pass Accuracy (%)")
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "4_pass_accuracy_boxplot.png"), dpi=300
)
plt.show()

# Chart 5: Goals vs. Assists Breakdown by Position
plt.figure(figsize=(9, 5))
x = np.arange(len(pos_means.index))
width = 0.35
plt.bar(
    x - width / 2, pos_means["goals"], width, label="Goals", color="#4C72B0"
)
plt.bar(
    x + width / 2, pos_means["assists"], width, label="Assists", color="#DD8452"
)
plt.xticks(x, pos_means.index)
plt.title("Average Goals and Assists by Position")
plt.xlabel("Position")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "5_goals_vs_assists.png"), dpi=300)
plt.show()

# Chart 6: Metric Correlation Heatmap
numeric_cols = [
    "age",
    "matches_played",
    "minutes_played",
    "pass_accuracy_pct",
    "goals",
    "assists",
    "shots_on_target",
]
valid_cols = [c for c in numeric_cols if c in df_clean.columns]
corr = df_clean[valid_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(valid_cols)), valid_cols, rotation=45, ha="right")
plt.yticks(range(len(valid_cols)), valid_cols)
plt.title("Metrics Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "6_correlation_heatmap.png"), dpi=300)
plt.show()