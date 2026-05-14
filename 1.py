import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['species'] = iris.target

print(df.head())

# Features and types
print("\nFeature Types:")
print(df.dtypes)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Additional statistics
print("\nVariance:")
print(df.var(numeric_only=True))

print("\nPercentiles:")
print(df.quantile([0.25, 0.5, 0.75], numeric_only=True))

# Histograms
df.hist(figsize=(10,8))
plt.suptitle("Histograms")
plt.show()

# Boxplots
df.plot(kind='box', subplots=False, figsize=(10,6))
plt.title("Boxplots")
plt.show()