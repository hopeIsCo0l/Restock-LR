import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("restocking_data.csv")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Summary statistics
print("\nStatistical Summary:")
print(df.describe())

# Count of restocking needs
print("\nRestocking Need Distribution:")
print(df["Restock_Needed"].value_counts())

# ------------------------------
# Univariate Analysis
# ------------------------------

# Set style for plots
sns.set_style("whitegrid")

# Histogram for Current Stock
plt.figure(figsize=(8, 5))
plt.hist(df["Current_Stock"], bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("Distribution of Current Stock", fontsize=14)
plt.xlabel("Current Stock", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Histogram for Average Sales
plt.figure(figsize=(8, 5))
plt.hist(df["Average_Sales"], bins=20, color="green", edgecolor="black", alpha=0.7)
plt.title("Distribution of Average Sales", fontsize=14)
plt.xlabel("Average Sales", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Countplot for Product Category
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Product_Category", palette="Set3")
plt.title("Frequency of Product Categories", fontsize=14)
plt.xlabel("Product Category", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# ------------------------------
# Bivariate Analysis
# ------------------------------

# Box plot: Current Stock vs Restock Needed
plt.figure(figsize=(8, 5))
sns.boxplot(x="Restock_Needed", y="Current_Stock", data=df, palette="Set2")
plt.title("Current Stock vs Restock Needed", fontsize=14)
plt.xlabel("Restock Needed (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Current Stock", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Box plot: Average Sales vs Restock Needed
plt.figure(figsize=(8, 5))
sns.boxplot(x="Restock_Needed", y="Average_Sales", data=df, palette="Set2")
plt.title("Average Sales vs Restock Needed", fontsize=14)
plt.xlabel("Restock Needed (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Average Sales", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 5))
correlation_matrix = df[["Current_Stock", "Average_Sales", "Days_Since_Last_Restock", "Restock_Needed"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=14)
plt.show()
