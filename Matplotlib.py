import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Streamlit app title
st.title("Mobile Sales Data Analysis Tool")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:")
    st.dataframe(data.head(100))

    # Handle missing values
    st.write("### Handling Missing Values")
    if data.isnull().values.any():
        st.write("Missing Values Found!")
        data = data.fillna(0)  # Replace missing values with 0 (can be customized)
        st.write("Missing values filled with 0.")
    else:
        st.write("No missing values found.")

    # Basic statistics
    st.write("### Basic Statistics")
    st.write(data.describe())

    # Visualization: Distribution of Units Sold
    st.write("### Distribution of Units Sold")
    plt.figure(figsize=(8, 5))
    sns.histplot(data['Units Sold (million )'], kde=True, bins=20)
    st.pyplot(plt)

    # Visualization: Trend of Units Sold Over Years
    st.write("### Trend of Units Sold Over Years")
    if 'Year' in data.columns:
        yearly_data = data.groupby('Year')['Units Sold (million )'].sum().reset_index()
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='Year', y='Units Sold (million )', data=yearly_data, marker='o')
        plt.title("Trend of Units Sold Over Years")
        st.pyplot(plt)
    else:
        st.write("Year column not found in the dataset.")

    # Statistical Analysis: T-test
    st.write("### T-Test Analysis")
    if 'Smartphone?' in data.columns:
        smartphones = data[data['Smartphone?'] == True]['Units Sold (million )']
        non_smartphones = data[data['Smartphone?'] == False]['Units Sold (million )']
        t_stat, p_value = ttest_ind(smartphones, non_smartphones)
        st.write(f"T-statistic: {t_stat}, P-value: {p_value}")
    else:
        st.write("Smartphone? column not found in the dataset.")

    # Correlation Matrix
    st.write("### Correlation Matrix")
    plt.figure(figsize=(10, 6))
    # correlation_matrix = data.corr()
    data_numeric = data.select_dtypes(include=['number'])
    correlation_matrix = data_numeric.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # Data Manipulation: Top Manufacturers
    st.write("### Top 5 Manufacturers by Total Units Sold")
    if 'Manufacturer' in data.columns:
        manufacturer_data = data.groupby('Manufacturer')['Units Sold (million )'].sum().reset_index()
        manufacturer_data = manufacturer_data.sort_values(by='Units Sold (million )', ascending=False).head(5)
        st.dataframe(manufacturer_data)
    else:
        st.write("Manufacturer column not found in the dataset.")

    # Bar Chart: Top 10 Best-Selling Models
    st.write("### Top 10 Best-Selling Mobile Phone Models")
    if 'Model' in data.columns:
        top_models = data.groupby('Model')['Units Sold (million )'].sum().reset_index()
        top_models = top_models.sort_values(by='Units Sold (million )', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Units Sold (million )', y='Model', data=top_models, palette="viridis")
        plt.title("Top 10 Best-Selling Mobile Phone Models")
        st.pyplot(plt)
    else:
        st.write("Model column not found in the dataset.")

    # Box Plot: Smartphones vs Non-Smartphones
    st.write("### Units Sold: Smartphones vs Non-Smartphones")
    if 'Smartphone?' in data.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='Smartphone?', y='Units Sold (million )', data=data)
        plt.title("Units Sold: Smartphones vs Non-Smartphones")
        st.pyplot(plt)
    else:
        st.write("Smartphone? column not found in the dataset.")

    # Machine Learning: Linear Regression
    st.write("### Machine Learning: Linear Regression")
    if 'Year' in data.columns:
        X = data[['Year']]
        y = data['Units Sold (million )']
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Plotting the regression
        plt.figure(figsize=(8, 5))
        plt.scatter(data['Year'], data['Units Sold (million )'], color='blue', label='Actual Data')
        plt.plot(data['Year'], predictions, color='red', label='Regression Line')
        plt.title("Linear Regression: Units Sold vs Year")
        plt.legend()
        st.pyplot(plt)

        # Model Evaluation
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        st.write(f"R-squared: {r2}")
        st.write(f"Mean Squared Error: {mse}")
    else:
        st.write("Year column not found in the dataset.")
