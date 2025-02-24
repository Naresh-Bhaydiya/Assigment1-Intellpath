import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from IPython.display import display

# Load dataset from Google Drive URL
DATA_URL = 'https://drive.google.com/uc?id=1Lff4UcvEu3cdCxWwSy-sRRIzHXiIqjnt'

def load_data(url):
    try:
        data = pd.read_csv(url)
        print("Dataset successfully loaded.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

data = load_data(DATA_URL)
if data is None:
    sys.exit("Exiting: Failed to load data.")

# Display dataset preview
def preview_data(df, rows=10):
    print("\n### Dataset Preview:")
    display(df.head(rows))

preview_data(data, rows=100)

# Handle missing values
def handle_missing_values(df):
    print("\n### Handling Missing Values")
    if df.isnull().values.any():
        df.fillna(0, inplace=True)
        print("Missing values found and filled with 0.")
    else:
        print("No missing values found.")

handle_missing_values(data)

# Basic statistics
def display_statistics(df):
    print("\n### Basic Statistics")
    display(df.describe())

display_statistics(data)

# Visualization: Distribution of Units Sold
def plot_units_distribution(df):
    print("\n### Distribution of Units Sold")
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Units Sold (million )'], kde=True, bins=20)
    plt.show()

plot_units_distribution(data)

# Visualization: Trend of Units Sold Over Years
def plot_units_trend(df):
    print("\n### Trend of Units Sold Over Years")
    if 'Year' in df.columns:
        yearly_data = df.groupby('Year')['Units Sold (million )'].sum().reset_index()
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='Year', y='Units Sold (million )', data=yearly_data, marker='o')
        plt.title("Trend of Units Sold Over Years")
        plt.show()
    else:
        print("Year column not found in the dataset.")

plot_units_trend(data)

# Statistical Analysis: T-test
def perform_ttest(df):
    print("\n### T-Test Analysis")
    if 'Smartphone?' in df.columns:
        smartphones = df[df['Smartphone?'] == True]['Units Sold (million )']
        non_smartphones = df[df['Smartphone?'] == False]['Units Sold (million )']
        t_stat, p_value = ttest_ind(smartphones, non_smartphones, nan_policy='omit')
        print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
    else:
        print("Smartphone? column not found in the dataset.")

perform_ttest(data)

# Correlation Matrix
def plot_correlation_matrix(df):
    print("\n### Correlation Matrix")
    plt.figure(figsize=(10, 6))
    numeric_data = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.show()

plot_correlation_matrix(data)

# Data Manipulation: Top Manufacturers
def top_manufacturers(df):
    print("\n### Top 5 Manufacturers by Total Units Sold")
    if 'Manufacturer' in df.columns:
        manufacturer_data = df.groupby('Manufacturer')['Units Sold (million )'].sum().reset_index()
        top_manufacturers = manufacturer_data.sort_values(by='Units Sold (million )', ascending=False).head(5)
        display(top_manufacturers)
    else:
        print("Manufacturer column not found in the dataset.")

top_manufacturers(data)

# Bar Chart: Top 10 Best-Selling Models
def top_models_chart(df):
    print("\n### Top 10 Best-Selling Mobile Phone Models")
    if 'Model' in df.columns:
        top_models = df.groupby('Model')['Units Sold (million )'].sum().reset_index()
        top_models = top_models.sort_values(by='Units Sold (million )', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Units Sold (million )', y='Model', data=top_models, palette="viridis")
        plt.title("Top 10 Best-Selling Mobile Phone Models")
        plt.show()
    else:
        print("Model column not found in the dataset.")

top_models_chart(data)

# Box Plot: Smartphones vs Non-Smartphones
def plot_smartphone_sales(df):
    print("\n### Units Sold: Smartphones vs Non-Smartphones")
    if 'Smartphone?' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='Smartphone?', y='Units Sold (million )', data=df)
        plt.title("Units Sold: Smartphones vs Non-Smartphones")
        plt.show()
    else:
        print("Smartphone? column not found in the dataset.")

plot_smartphone_sales(data)

# Machine Learning: Linear Regression
def perform_linear_regression(df):
    print("\n### Machine Learning: Linear Regression")
    if 'Year' in df.columns:
        X = df[['Year']]
        y = df['Units Sold (million )']
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(df['Year'], y, color='blue', label='Actual Data')
        plt.plot(df['Year'], predictions, color='red', label='Regression Line')
        plt.title("Linear Regression: Units Sold vs Year")
        plt.legend()
        plt.show()

        print(f"R-squared: {r2_score(y, predictions):.3f}")
        print(f"Mean Squared Error: {mean_squared_error(y, predictions):.3f}")
    else:
        print("Year column not found in the dataset.")

perform_linear_regression(data)

print("\nAnalysis Completed Successfully.")
