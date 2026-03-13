#Part 1
import pandas as pd
import numpy as np
df = pd.read_csv("winequality-white.csv", sep=';')
print(df.shape)
print(df.head())
print(df.info())
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.sort_values(ascending=False))

#Part 2
target_col = "quality"
# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
# Remove target from numeric predictors
if target_col in numeric_cols:
    numeric_cols.remove(target_col)
# Fill numeric missing values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
# Fill categorical missing values with mode
for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])
# One-hot encoding
df = pd.get_dummies(df, drop_first=True)
# Define X and y
X = df.drop("quality", axis=1)
y = df["quality"]
# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Part 3
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
explained_variance = S**2 / np.sum(S**2)
cumulative_variance = np.cumsum(explained_variance)
# Choose k to retain at least 90% variance
k = np.argmax(cumulative_variance >= 0.90) + 1
X_reduced = U[:, :k] @ np.diag(S[:k])

#Explained Variance Plot
import matplotlib.pyplot as plt
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.axhline(y=0.90, linestyle='--')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance Plot")
plt.grid(True)
plt.show()