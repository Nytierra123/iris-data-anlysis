import pandas as pd

# Load the CSV file
df = pd.read_csv("1) iris.csv")

# View the first 5 rows
print(df.head())


# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Standardize the 'species' column
df['species'] = df['species'].str.strip().str.lower()

# Final summary
print("\nCleaned data preview:\n", df.head())
print("\nUnique species:\n", df['species'].unique())
print("\nNew shape:", df.shape)


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Statistical summary
print("\nSummary statistics:\n", df.describe())

# 2. Distribution of each numeric feature
df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# 3. Boxplots to compare features across species
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='species', y='sepal_length')
plt.title("Sepal Length by Species")
plt.show()

# 4. Pair plot for all features
sns.pairplot(df, hue='species')
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()

# Bar graph
sns.countplot(x='species', data=df)
plt.title("Count of Each Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# line graph
plt.plot(df.index, df['sepal_length'])
plt.title("Sepal Length Across Samples")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.show()

#scatter graph
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df)
plt.title("Petal Length vs Width by Species")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(title='Species')
plt.show()




