# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("DATA ANALYSIS ASSIGNMENT: PANDAS & MATPLOTLIB")
print("=" * 60)

# ==================== TASK 1: LOAD AND EXPLORE DATASET ====================
print("\nTASK 1: LOAD AND EXPLORE THE DATASET")
print("-" * 40)

try:
    # Load the Iris dataset
    iris = load_iris()
    
    # Create a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✓ Dataset loaded successfully!")
    print(f"Dataset source: Iris dataset from scikit-learn")
    
except Exception as e:
    print(f" Error loading dataset: {e}")
    # Fallback: create sample data if iris dataset fails
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150),
        'petal length (cm)': np.random.normal(3.8, 1.8, 150),
        'petal width (cm)': np.random.normal(1.2, 0.8, 150),
        'species': np.repeat([0, 1, 2], 50),
        'species_name': np.repeat(['setosa', 'versicolor', 'virginica'], 50)
    })
    print("✓ Sample data created as fallback!")

# Display first few rows
print(f"\nDataset Shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore data structure
print(f"\nDataset Info:")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print(f"\nData Types:")
print(df.dtypes)

# Check for missing values
print(f"\nMissing Values Check:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing_values[missing_values > 0])
    print("\nCleaning data by dropping missing values...")
    df = df.dropna()
    print(f"✓ Dataset cleaned. New shape: {df.shape}")

# ==================== TASK 2: BASIC DATA ANALYSIS ====================
print("\n" + "=" * 60)
print("TASK 2: BASIC DATA ANALYSIS")
print("-" * 40)

# Basic statistics for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
print("Basic Statistics for Numerical Columns:")
print(df[numerical_cols].describe().round(3))

# Group analysis by species
print(f"\nGrouped Analysis by Species:")
species_analysis = df.groupby('species_name')[numerical_cols].agg(['mean', 'std']).round(3)
print(species_analysis)

# Additional analysis - correlation matrix
print(f"\nCorrelation Analysis:")
correlation_matrix = df[numerical_cols].corr().round(3)
print(correlation_matrix)

# Key findings
print(f"\n KEY FINDINGS FROM ANALYSIS:")
print(f"1. Dataset contains {len(df)} samples with {len(df.columns)} features")
print(f"2. Three species: {', '.join(df['species_name'].unique())}")
print(f"3. Highest correlation: {correlation_matrix.abs().unstack().sort_values(ascending=False).iloc[1]:.3f}")

# ==================== TASK 3: DATA VISUALIZATION ====================
print("\n" + "=" * 60)
print("TASK 3: DATA VISUALIZATION")
print("-" * 40)

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Iris Dataset Analysis - Four Types of Visualizations', fontsize=16, fontweight='bold')

# 1. LINE CHART - Trend over sample index (simulating time series)
plt.subplot(2, 2, 1)
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    plt.plot(species_data.index, species_data['sepal length (cm)'], 
             marker='o', markersize=3, alpha=0.7, label=species)

plt.title('Line Chart: Sepal Length Trends Across Samples', fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. BAR CHART - Average measurements by species
plt.subplot(2, 2, 2)
species_means = df.groupby('species_name')['petal length (cm)'].mean()
bars = plt.bar(species_means.index, species_means.values, 
               color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.title('Bar Chart: Average Petal Length by Species', fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=45)

# 3. HISTOGRAM - Distribution of sepal width
plt.subplot(2, 2, 3)
plt.hist(df['sepal width (cm)'], bins=20, color='purple', alpha=0.7, edgecolor='black')
plt.title('Histogram: Distribution of Sepal Width', fontweight='bold')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Add statistics text
mean_val = df['sepal width (cm)'].mean()
std_val = df['sepal width (cm)'].std()
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
plt.legend()

# 4. SCATTER PLOT - Relationship between sepal length and petal length
plt.subplot(2, 2, 4)
colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
                c=colors[species], label=species, alpha=0.7, s=50)

plt.title('Scatter Plot: Sepal Length vs Petal Length', fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.show()

# ==================== ADDITIONAL ADVANCED VISUALIZATIONS ====================
print("\nCREATING ADDITIONAL ADVANCED VISUALIZATIONS...")

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            mask=mask, square=True, fmt='.3f')
plt.title('Correlation Heatmap of Iris Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Box plots for all features by species
plt.figure(figsize=(15, 10))
features = [col for col in df.columns if 'cm)' in col]

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='species_name', y=feature, palette='Set2')
    plt.title(f'Box Plot: {feature} by Species', fontweight='bold')
    plt.xticks(rotation=45)

plt.suptitle('Distribution of All Features by Species', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ==================== SUMMARY OF FINDINGS ====================
print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS AND OBSERVATIONS")
print("=" * 60)

print("DATASET OVERVIEW:")
print(f"   • Total samples: {len(df)}")
print(f"   • Features analyzed: 4 morphological measurements")
print(f"   • Species: 3 (Setosa, Versicolor, Virginica)")

print("\n KEY STATISTICAL INSIGHTS:")
print(f"   • Petal length shows highest variation (std: {df['petal length (cm)'].std():.3f})")
print(f"   • Sepal width shows lowest variation (std: {df['sepal width (cm)'].std():.3f})")
print(f"   • Strongest correlation: Petal length & Petal width ({correlation_matrix.loc['petal length (cm)', 'petal width (cm)']:.3f})")

print("\nSPECIES CHARACTERISTICS:")
setosa_avg = df[df['species_name'] == 'setosa']['petal length (cm)'].mean()
virginica_avg = df[df['species_name'] == 'virginica']['petal length (cm)'].mean()
print(f"   • Setosa has smallest petals (avg length: {setosa_avg:.2f} cm)")
print(f"   • Virginica has largest petals (avg length: {virginica_avg:.2f} cm)")
print(f"   • Clear species separation visible in scatter plots")

print("\nVISUALIZATION INSIGHTS:")
print("   • Line chart reveals species clustering patterns")
print("   • Bar chart shows clear differences in petal lengths")
print("   • Histogram shows normal distribution of sepal width")
print("   • Scatter plot demonstrates strong species separability")

print("\n ASSIGNMENT COMPLETION STATUS:")
print("   ✓ Task 1: Dataset loaded and explored successfully")
print("   ✓ Task 2: Statistical analysis completed with key findings")
print("   ✓ Task 3: All four required visualizations created")
print("   ✓ Additional: Advanced visualizations and comprehensive analysis")

print("\n" + "=" * 60)
print("END OF ANALYSIS")
print("=" * 60)