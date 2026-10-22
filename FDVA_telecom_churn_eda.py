"""
FRAMEWORK FOR DATA AND VISUAL ANALYTICS - MINI PROJECT
Business Analytics: Exploratory Data Analysis (EDA) of Telecom Customer Churn
Dataset : IBM Telco Customer Churn (7,043 customers, 20 attributes + churn result)
          A public sample dataset released by IBM, widely used on Kaggle.
          This script downloads a public GitHub copy of the same CSV.
Libraries: Pandas, NumPy, Matplotlib

====================================================================
1. AIM OF THE PROJECT
====================================================================
This project performs Exploratory Data Analysis (EDA) on a public telecom dataset of 7,043
customers to understand which customer characteristics (contract type, tenure, charges,
services) are linked to customer churn (customers leaving the company). It cleans the data
(hidden missing values, duplicates, outliers), builds a new feature, computes statistics and
correlations, and uses Matplotlib visualizations to present the key findings.

====================================================================
2. PROCEDURE
====================================================================
1. Import the required libraries.
2. Load the dataset (from a public URL, or a local churn.csv if you have one).
3. Inspect the dataset (shape, data types, first rows, summary statistics).
4. Handle missing values: find the hidden missing values (blank text in TotalCharges)
   and fill them with the median.
5. Remove duplicate records.
6. Perform data cleaning (consistent data types, feature engineering, final check).
7. Detect and analyze outliers using the IQR method and box plots.
8. Perform descriptive analysis (mean, median, skewness, comparison by churn).
9. Analyze relationships between variables (scatter plots, grouped bar and line charts).
10. Calculate correlations and draw a correlation heatmap.
11. Create visualizations (bar chart, histograms, box plots, scatter plots, heatmap, line charts).
12. Interpret the results and print the findings.

====================================================================
3. CODE
====================================================================
"""

# ---------------------------------------------------------------
# Step 1: Import required libraries
# ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 25)

# Two colours used in every chart: green = customer stayed, red = customer churned
COLOR_STAY, COLOR_CHURN = "#2e8b57", "#d1402f"


def heading(text):
    """Print a clear section heading in the console output."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------
# Step 2: Load the dataset
# ---------------------------------------------------------------
# Public copy of the dataset (no login needed). A local 'churn.csv' is used first
# if it exists next to this script, which is handy when you are offline.
DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
]


def load_data():
    if os.path.exists("churn.csv"):
        print("Loading local file: churn.csv")
        return pd.read_csv("churn.csv")
    for url in DATA_URLS:
        try:
            print("Downloading:", url)
            return pd.read_csv(url)
        except Exception as err:
            print("  failed:", type(err).__name__)
    raise SystemExit("Could not download the dataset. Download 'Telco-Customer-Churn.csv' "
                     "(IBM Telco Customer Churn) from Kaggle, save it as 'churn.csv' next to "
                     "this script, then run again.")


df = load_data()
n_raw_cols = df.shape[1]   # columns in the original file

# ---------------------------------------------------------------
# Step 3: Inspect the dataset
# ---------------------------------------------------------------
heading("STEP 3: DATASET INSPECTION")
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn data types and non-null counts:")
df.info()
print("\nSummary statistics of the raw numeric data:")
print(df.describe().T)

# Meaning of the main columns (for reference):
#   tenure = months the customer has stayed with the company
#   MonthlyCharges = amount billed per month, TotalCharges = total amount billed so far
#   Contract = Month-to-month / One year / Two year
#   InternetService = DSL / Fiber optic / No
#   SeniorCitizen = 1 if the customer is a senior citizen, otherwise 0
#   Churn = Yes if the customer left the company in the last month, otherwise No

print("\nChurn counts:")
print(df["Churn"].value_counts())

# ---------------------------------------------------------------
# Step 4: Handle missing values
# ---------------------------------------------------------------
heading("STEP 4: MISSING VALUES")
print("Standard missing values (NaN) in the raw data:", int(df.isnull().sum().sum()))

# This dataset has no NaN cells, BUT 'TotalCharges' is stored as TEXT (object type) because
# a few cells contain a blank space " " instead of a number. These blanks are really
# missing values. pd.to_numeric(errors="coerce") turns anything that is not a number into NaN.
blank_mask = df["TotalCharges"].astype(str).str.strip() == ""
hidden_missing = int(blank_mask.sum())
print("\nHidden missing values (blank text in TotalCharges):", hidden_missing,
      "(%.2f%% of rows)" % (hidden_missing / len(df) * 100))
print("Tenure of those customers (they are brand-new customers, never billed yet):")
print(df.loc[blank_mask, "tenure"].value_counts())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Create a numeric target column: 1 = customer churned, 0 = customer stayed
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

missing_cols = ["TotalCharges"]

# Keep a copy that still has the missing values (NaN). The outlier checks, histograms,
# group statistics and correlations below use only the REAL values (Pandas skips NaN
# automatically), while 'df' becomes the fully cleaned, filled dataset.
df_observed = df.copy()

# Chart 1: hidden missing values + class balance
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
miss_counts = df_observed[["tenure", "MonthlyCharges", "TotalCharges"]].isnull().sum()
bars = ax[0].bar(miss_counts.index, miss_counts.values, color="#7a7a7a")
ax[0].bar_label(bars)
ax[0].set_title("Missing Values per Numeric Column (after fixing text blanks)")
ax[0].set_ylabel("Number of rows")
ax[0].set_ylim(0, max(miss_counts.max() * 1.4, 5))

counts = df["ChurnFlag"].value_counts().sort_index()
bars = ax[1].bar(["Stayed (0)", "Churned (1)"], counts.values, color=[COLOR_STAY, COLOR_CHURN])
ax[1].bar_label(bars)
ax[1].set_title("Class Distribution of Churn")
ax[1].set_ylabel("Number of customers")
plt.tight_layout()
plt.show()

# Fill missing values with the MEDIAN. The median is used (not the mean) because
# TotalCharges is strongly right-skewed and the median is not affected by extreme values.
medians = df[missing_cols].median()
print("\nMedian value used to fill the missing entries:")
print(medians.round(2))
df[missing_cols] = df[missing_cols].fillna(medians)
print("\nMissing values after filling:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------
# Step 5: Remove duplicate records
# ---------------------------------------------------------------
heading("STEP 5: DUPLICATE RECORDS")
n_dup_rows = int(df.duplicated().sum())
n_dup_ids = int(df["customerID"].duplicated().sum())
print("Fully duplicated rows found:", n_dup_rows)
print("Repeated customer IDs found:", n_dup_ids)
df = df.drop_duplicates().drop_duplicates(subset="customerID").reset_index(drop=True)
print("Rows after the duplicate check:", len(df))

# ---------------------------------------------------------------
# Step 6: Data cleaning (data types, new feature, final check)
# ---------------------------------------------------------------
heading("STEP 6: DATA CLEANING SUMMARY")
# Remove stray spaces in all text columns
text_cols = df.select_dtypes(include="object").columns
for c in text_cols:
    df[c] = df[c].str.strip()

# Columns that are whole numbers by nature are stored as integers
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
df["tenure"] = df["tenure"].astype(int)
df["ChurnFlag"] = df["ChurnFlag"].astype(int)

# Feature engineering: count how many optional services each customer uses
service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
df_observed["NumServices"] = (df_observed[service_cols] == "Yes").sum(axis=1)

print("Final shape:", df.shape)
print("Missing values left:", int(df.isnull().sum().sum()), "| Duplicates left:", int(df.duplicated().sum()))
print("New feature 'NumServices' (0 to 8 optional services used per customer):")
print(df["NumServices"].describe().round(2))

num_features = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]

# ---------------------------------------------------------------
# Step 7: Detect and analyze outliers (IQR method)
# ---------------------------------------------------------------
heading("STEP 7: OUTLIERS (IQR METHOD)")
# A value is an outlier if it is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
# Only real values are used (missing values are excluded).
rows = []
for col in num_features:
    values = df_observed[col].dropna()
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((values < low) | (values > high)).sum())
    rows.append([col, round(low, 2), round(high, 2), n_out, round(n_out / len(values) * 100, 1)])
outlier_table = pd.DataFrame(rows, columns=["Column", "Lower limit", "Upper limit", "Outliers", "Percent"])
print(outlier_table.to_string(index=False))

# Chart 2: box plots of every numeric feature (dots outside the whiskers are outliers)
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
for ax, col in zip(axes, num_features):
    bp = ax.boxplot(df_observed[col].dropna(), patch_artist=True, widths=0.5,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=COLOR_CHURN,
                                    markeredgecolor=COLOR_CHURN))
    bp["boxes"][0].set_facecolor("#a9c7e8")
    bp["medians"][0].set_color("black")
    ax.set_title(col, fontsize=10)
    ax.set_xticks([])
fig.suptitle("Box Plots of Numeric Features (red dots = outliers)", fontsize=13)
plt.tight_layout()
plt.show()

# Handling outliers: any outliers here would be real billing values, not typing errors, so
# they are KEPT. To show what they do, we compare the data with a "capped" copy where
# extreme values are clipped to the IQR limits.
df_capped = df_observed.copy()
for _, r in outlier_table.iterrows():
    df_capped[r["Column"]] = df_capped[r["Column"]].clip(r["Lower limit"], r["Upper limit"])
comparison = pd.DataFrame({
    "Mean (original)": df_observed[num_features].mean(),
    "Mean (capped)": df_capped[num_features].mean(),
    "Std (original)": df_observed[num_features].std(),
    "Std (capped)": df_capped[num_features].std(),
}).round(2)
print("\nEffect of capping outliers on mean and standard deviation:")
print(comparison)
if outlier_table["Outliers"].sum() == 0:
    print("\nNo IQR outliers were found, so capping changes nothing: the numeric columns are "
          "clean and no outlier treatment is needed.")

# ---------------------------------------------------------------
# Step 8: Descriptive analysis
# ---------------------------------------------------------------
heading("STEP 8: DESCRIPTIVE STATISTICS")
print("Summary statistics of the real values ('count' = values that were not missing):")
print(df_observed[num_features + ["SeniorCitizen", "ChurnFlag"]].describe().T.round(2))

print("\nSkewness (0 = symmetric, > 1 = strongly right-skewed):")
print(df_observed[num_features].skew().round(2))

print("\nMean of each numeric feature by Churn:")
print(df_observed.groupby("Churn")[num_features].mean().round(2).T)
print("\nMedian of each numeric feature by Churn:")
print(df_observed.groupby("Churn")[num_features].median().round(2).T)

# Chart 3: histograms (distribution of every numeric feature)
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
for ax, col in zip(axes, num_features):
    values = df_observed[col].dropna()
    ax.hist(values, bins=20, color="#6a9fd4", edgecolor="white")
    ax.axvline(values.mean(), color="black", linestyle="--", linewidth=1.2,
               label="mean = %.1f" % values.mean())
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
fig.suptitle("Histograms of Numeric Features (real values, missing excluded)", fontsize=13)
plt.tight_layout()
plt.show()

# Chart 4: box plots split by Churn for the numeric features
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
for ax, col in zip(axes, num_features):
    data = [df_observed.loc[df_observed["ChurnFlag"] == 0, col].dropna(),
            df_observed.loc[df_observed["ChurnFlag"] == 1, col].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="grey",
                                    markeredgecolor="grey"))
    for patch, colour in zip(bp["boxes"], [COLOR_STAY, COLOR_CHURN]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Stayed", "Churned"])
    ax.set_title(col)
fig.suptitle("Numeric Features: Customers Who Stayed vs Churned", fontsize=13)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Step 9 & 10: Relationships and correlations
# ---------------------------------------------------------------
heading("STEP 10: CORRELATION ANALYSIS")
# Pearson correlation: +1 = move together, -1 = move opposite, 0 = no linear relationship.
# Calculated on real values (each pair uses the rows where both values exist).
corr_cols = num_features + ["SeniorCitizen", "ChurnFlag"]
corr = df_observed[corr_cols].corr()
print(corr.round(2))

# Chart 5: correlation heatmap (drawn with Matplotlib imshow)
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        value = corr.values[i, j]
        ax.text(j, i, "%.2f" % value, ha="center", va="center", fontsize=9,
                color="white" if abs(value) > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Correlation coefficient")
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Chart 6: which feature is most related to churn? (negative = more of it, less churn)
corr_outcome = corr["ChurnFlag"].drop("ChurnFlag").sort_values()
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(corr_outcome.index, corr_outcome.values,
               color=[COLOR_CHURN if v > 0 else COLOR_STAY for v in corr_outcome.values])
ax.bar_label(bars, fmt="%.2f", padding=3)
ax.axvline(0, color="black", linewidth=0.8)
limit = corr_outcome.abs().max() + 0.1
ax.set_xlim(-limit, limit)
ax.set_xlabel("Correlation with Churn (red = more churn, green = less churn)")
ax.set_title("Correlation of Each Feature with Churn")
plt.tight_layout()
plt.show()

# Chart 7: scatter plots coloured by churn
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for flag, colour, label in [(0, COLOR_STAY, "Stayed"), (1, COLOR_CHURN, "Churned")]:
    part = df[df["ChurnFlag"] == flag]
    ax[0].scatter(part["tenure"], part["MonthlyCharges"], c=colour, alpha=0.35, s=14, label=label)
    ax[1].scatter(part["tenure"], part["TotalCharges"], c=colour, alpha=0.35, s=14, label=label)
ax[0].set_xlabel("Tenure (months)"); ax[0].set_ylabel("Monthly charges"); ax[0].set_title("Monthly Charges vs Tenure")
ax[1].set_xlabel("Tenure (months)"); ax[1].set_ylabel("Total charges"); ax[1].set_title("Total Charges vs Tenure")
ax[0].legend(); ax[1].legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Step 11: Group-wise analysis with bar charts and line charts
# ---------------------------------------------------------------
heading("STEP 11: CHURN RATE BY GROUP")
# Group the continuous columns into categories so patterns are easy to read
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 36, 48, 60, 72],
                           labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
df["ChargeGroup"] = pd.cut(df["MonthlyCharges"], bins=[0, 35, 70, 90, 150],
                           labels=["Low (<=35)", "Medium (35-70)", "High (70-90)", "Very high (90+)"])


def rate_table(group_col):
    """Percentage of churned customers and number of customers in each group."""
    g = df.groupby(group_col, observed=True)["ChurnFlag"]
    return pd.DataFrame({"customers": g.count(), "churn_rate_%": (g.mean() * 100).round(1)})


contract_rate = rate_table("Contract")
internet_rate = rate_table("InternetService")
payment_rate = rate_table("PaymentMethod")
tenure_rate = rate_table("TenureGroup")
charge_rate = rate_table("ChargeGroup")
for name, table in [("Contract type", contract_rate), ("Internet service", internet_rate),
                    ("Payment method", payment_rate), ("Tenure group (months)", tenure_rate),
                    ("Monthly charge group", charge_rate)]:
    print("\nChurn rate by", name)
    print(table)

# Chart 8: bar charts - churn rate by contract type and by internet service
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, table, title in [(ax[0], contract_rate, "Churn Rate by Contract Type"),
                        (ax[1], internet_rate, "Churn Rate by Internet Service")]:
    labels = ["%s\n(n=%d)" % (idx, n) for idx, n in zip(table.index, table["customers"])]
    bars = a.bar(labels, table["churn_rate_%"], color="#6a9fd4", edgecolor="white")
    a.bar_label(bars, fmt="%.1f%%")
    a.set_title(title)
    a.set_ylabel("Customers who churned (%)")
    a.set_ylim(0, max(table["churn_rate_%"]) + 15)
plt.tight_layout()
plt.show()

# Chart 9: line charts - trend over groups
mean_charge = df.groupby(["TenureGroup", "Churn"], observed=True)["MonthlyCharges"].mean().unstack()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
labels = ["%s\n(n=%d)" % (idx, n) for idx, n in zip(tenure_rate.index, tenure_rate["customers"])]
ax[0].plot(labels, tenure_rate["churn_rate_%"], marker="o", color=COLOR_CHURN, linewidth=2)
ax[0].set_title("Churn Rate vs Customer Tenure")
ax[0].set_ylabel("Customers who churned (%)")
ax[0].set_xlabel("Tenure group (months)")
ax[0].grid(alpha=0.3)

ax[1].plot(mean_charge.index.astype(str), mean_charge["No"], marker="o", color=COLOR_STAY,
           linewidth=2, label="Stayed")
ax[1].plot(mean_charge.index.astype(str), mean_charge["Yes"], marker="o", color=COLOR_CHURN,
           linewidth=2, label="Churned")
ax[1].set_title("Average Monthly Charges by Tenure Group")
ax[1].set_ylabel("Mean monthly charges")
ax[1].set_xlabel("Tenure group (months)")
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Step 12: Interpret the results (all numbers below are computed from the data)
# ---------------------------------------------------------------
heading("STEP 12: KEY INSIGHTS FROM THE EDA")

churn_pct = df["ChurnFlag"].mean() * 100
top_pos = corr_outcome.sort_values(ascending=False).iloc[0]
top_pos_name = corr_outcome.sort_values(ascending=False).index[0]
top_neg_name = corr_outcome.sort_values().index[0]
top_neg = corr_outcome.sort_values().iloc[0]
med_tenure = df_observed.groupby("Churn")["tenure"].median()
med_monthly = df_observed.groupby("Churn")["MonthlyCharges"].median()
med_services = df_observed.groupby("Churn")["NumServices"].median()

# strongest relationship between two numeric features (ignoring Churn)
feat_corr = corr.loc[num_features, num_features]
pairs = feat_corr.where(np.triu(np.ones(feat_corr.shape, dtype=bool), k=1)).stack()
best_pair = pairs.abs().idxmax()

worst_outlier_col = outlier_table.sort_values("Outliers", ascending=False).iloc[0]

# (very small groups are left out of the best/worst comparison)
contract_big = contract_rate[contract_rate["customers"] >= 20]["churn_rate_%"]
internet_big = internet_rate[internet_rate["customers"] >= 20]["churn_rate_%"]
payment_big = payment_rate[payment_rate["customers"] >= 20]["churn_rate_%"]
tenure_big = tenure_rate[tenure_rate["customers"] >= 20]["churn_rate_%"]

print("1. Dataset: %d customers, %d columns. %.1f%% churned, so the classes are "
      "imbalanced (about 1 in 4)." % (len(df), n_raw_cols, churn_pct))
print("2. Data quality: no NaN cells and %d duplicates, but TotalCharges was stored as text "
      "with %d blank values (new customers with tenure 0). They were converted to numbers "
      "and filled with the median." % (n_dup_rows + n_dup_ids, hidden_missing))
if worst_outlier_col["Outliers"] > 0:
    print("3. Outliers: %s has the most (%d values, %.1f%%). They are real billing values, so "
          "they were kept." % (worst_outlier_col["Column"], worst_outlier_col["Outliers"],
                               worst_outlier_col["Percent"]))
else:
    print("3. Outliers: none were found with the IQR method, so no outlier treatment was needed.")
print("4. Strongest positive correlation with churn: %s (r=%.2f). Strongest negative: %s (r=%.2f)."
      % (top_pos_name, top_pos, top_neg_name, top_neg))
print("5. Median tenure is %.0f months for churned vs %.0f for loyal customers; median monthly "
      "charge is %.1f vs %.1f; median number of services is %.0f vs %.0f." % (
          med_tenure["Yes"], med_tenure["No"], med_monthly["Yes"], med_monthly["No"],
          med_services["Yes"], med_services["No"]))
print("6. By contract type: highest churn is %s (%.1f%%), lowest is %s (%.1f%%)."
      % (contract_big.idxmax(), contract_big.max(), contract_big.idxmin(), contract_big.min()))
print("7. By internet service: highest churn is %s (%.1f%%), lowest is %s (%.1f%%). By payment "
      "method: highest is %s (%.1f%%)." % (internet_big.idxmax(), internet_big.max(),
                                           internet_big.idxmin(), internet_big.min(),
                                           payment_big.idxmax(), payment_big.max()))
print("8. By tenure: customers in their first months are the most likely to leave (%s months: "
      "%.1f%%), while churn falls to %.1f%% for %s months." % (
          tenure_big.idxmax(), tenure_big.max(), tenure_big.min(), tenure_big.idxmin()))
print("9. Strongest relationship between two numeric features: %s and %s (r=%.2f), so some "
      "features carry overlapping information." % (best_pair[0], best_pair[1], pairs[best_pair]))
print("\nBusiness suggestions: offer discounts to move month-to-month customers onto longer "
      "contracts, give extra support and onboarding in the first year, and review the "
      "fiber-optic and electronic-check customer segments that show high churn.")
print("\nLimitations: correlation does not prove causation; the dataset is a single snapshot "
      "of one company (a sample dataset), and 11 missing TotalCharges values were filled by "
      "the median, so the findings may not apply to other telecom companies.")
