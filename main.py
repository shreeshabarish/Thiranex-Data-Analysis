import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations folder
os.makedirs("visualizations", exist_ok=True)

# Load dataset
df = pd.read_csv("data/SampleSuperstore.csv", encoding="latin1")
# Clean the dataset

# Remove duplicate rows
df = df.drop_duplicates()

# Convert date columns
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# Save cleaned dataset
df.to_csv(
    "cleaned_data/Cleaned_SampleSuperstore.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")

# Show first 5 rows
print(df.head())

# Show number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Convert date columns to datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

# Check date data types
print("\nDate data types:")
print(df[["Order Date", "Ship Date"]].dtypes)

# Numerical summary
print("\nNumerical summary:")
print(df[["Sales", "Quantity", "Discount", "Profit"]].describe())

# IQR outlier analysis
Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)

IQR = Q3 - Q1

print("\nIQR Analysis:")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

# Calculate lower and upper limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

# Find sales outliers
outliers = df[
    (df["Sales"] < lower_limit) |
    (df["Sales"] > upper_limit)
]

print("Number of outliers:", len(outliers))

print("\nFirst 20 sales outliers:")
print(
    outliers[
        [
            "Order ID",
            "Product Name",
            "Sales",
            "Quantity",
            "Discount",
            "Profit"
        ]
    ].head(20)
)

# Top 10 sales
print("\nTop 10 sales:")
print(
    df[
        [
            "Order ID",
            "Product Name",
            "Sales",
            "Quantity",
            "Discount",
            "Profit"
        ]
    ]
    .sort_values("Sales", ascending=False)
    .head(10)
)

# Check unique values in categorical columns
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())

# Sales and Profit by Category
category_analysis = df.groupby("Category")[["Sales", "Profit"]].sum()

print("\nSales and Profit by Category:")
print(category_analysis)

# Sales and Profit by Sub-Category
subcategory_analysis = df.groupby("Sub-Category")[["Sales", "Profit"]].sum()

print("\nSales and Profit by Sub-Category:")
print(
    subcategory_analysis.sort_values(
        "Profit",
        ascending=False
    )
)

# Sales by Category
plt.figure(figsize=(8, 5))

plt.bar(
    category_analysis.index,
    category_analysis["Sales"]
)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/sales_by_category.png",
    dpi=300
)

plt.show()

# Profit by Category
plt.figure(figsize=(8, 5))

plt.bar(
    category_analysis.index,
    category_analysis["Profit"]
)

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig(
    "visualizations/profit_by_category.png",
    dpi=300
)

plt.show()

# Sales and Profit by Category
x = np.arange(len(category_analysis.index))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    category_analysis["Sales"],
    width,
    label="Sales"
)

plt.bar(
    x + width / 2,
    category_analysis["Profit"],
    width,
    label="Profit"
)

plt.title("Sales and Profit by Category")
plt.xlabel("Category")
plt.ylabel("Amount")

plt.xticks(
    x,
    category_analysis.index
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "visualizations/sales_profit_by_category.png",
    dpi=300
)

plt.show()

# Sales by Year
yearly_sales = df.groupby(
    df["Order Date"].dt.year
)["Sales"].sum()

print("\nSales by Year:")
print(yearly_sales)

# Sales by Year chart
plt.figure(figsize=(8, 5))

plt.plot(
    yearly_sales.index,
    yearly_sales.values,
    marker="o"
)

plt.title("Sales by Year")
plt.xlabel("Year")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    "visualizations/sales_by_year.png",
    dpi=300
)

plt.show()

# Sales vs Profit
print("\nStarting scatter plot...")

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Sales"],
    df["Profit"],
    s=20,
    alpha=0.5
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "visualizations/sales_vs_profit.png",
    dpi=300
)

plt.show()

# Sales Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["Sales"],
    bins=30
)

plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "visualizations/sales_distribution.png",
    dpi=300
)

plt.show()

# Profit Boxplot
plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Profit"]
)

plt.title("Profit Distribution and Outliers")
plt.xlabel("Profit")

plt.tight_layout()

plt.savefig(
    "visualizations/profit_boxplot.png",
    dpi=300
)

plt.show()

# Sales Boxplot
plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["Sales"]
)

plt.title("Sales Distribution and Outliers")
plt.xlabel("Sales")

plt.tight_layout()

plt.savefig(
    "visualizations/sales_boxplot.png",
    dpi=300
)

plt.show()

# Seaborn Sales by Category
plt.figure(figsize=(8, 5))

sns.barplot(
    x=category_analysis.index,
    y=category_analysis["Sales"]
)

plt.title("Sales by Category - Seaborn")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig(
    "visualizations/seaborn_sales_by_category.png",
    dpi=300
)

plt.show()

# Seaborn Profit by Category
plt.figure(figsize=(8, 5))

sns.barplot(
    x=category_analysis.index,
    y=category_analysis["Profit"]
)

plt.title("Profit by Category - Seaborn")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig(
    "visualizations/seaborn_profit_by_category.png",
    dpi=300
)

plt.show()

# Correlation Heatmap
correlation = df[
    ["Sales", "Quantity", "Discount", "Profit"]
].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png",
    dpi=300
)

plt.show()

# Project completed
print("\nNah I would win!")
