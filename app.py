import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("📊 Mobile Sales Data Analytics")

# Upload File
uploaded_file = st.file_uploader("Upload Mobile Sales File (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Rename columns to match expected format
    df.rename(columns={
        'Rank': 'rank',
        'Manufacturer': 'manufacturer',
        'Model': 'model_name',
        'Form Factor': 'form_factor',
        'Smartphone?': 'is_smartphone',
        'Year': 'year_of_release',
        'Units Sold (million)': 'Units Sold (million )'
    }, inplace=True)
    
    df['year_of_release'] = df['year_of_release'].astype(int)
    
    st.success("File Uploaded Successfully!")
    
    # Sidebar Options
    st.sidebar.header("Filters")
    year_filter = st.sidebar.selectbox("Select Year", sorted(df['year_of_release'].unique()), index=0)
    manufacturer_filter = st.sidebar.selectbox("Select Manufacturer", sorted(df['manufacturer'].unique()))
    
    # Top 5 Models per Year
    st.subheader(f"📌 Top 5 Best-Selling Models in {year_filter}")
    top_models = df[df['year_of_release'] == year_filter].nlargest(5, 'Units Sold (million )')
    st.dataframe(top_models[['model_name', 'manufacturer', 'Units Sold (million )']])
    
    # Top 3 Manufacturers
    st.subheader("🏆 Top 3 Manufacturers by Total Sales")
    top_manufacturers = df.groupby('manufacturer')['Units Sold (million )'].sum().nlargest(3)
    st.dataframe(top_manufacturers.reset_index())
    
    # Moving Average of Sales
    st.subheader(f"📈 Moving Average of {manufacturer_filter}'s Sales")
    df_manufacturer = df[df['manufacturer'] == manufacturer_filter]
    df_manufacturer['moving_avg'] = df_manufacturer['Units Sold (million )'].rolling(window=3).mean()
    st.line_chart(df_manufacturer[['year_of_release', 'moving_avg']].set_index('year_of_release'))
    
    # Line Chart - Trend of Units Sold Over Years
    st.subheader("📉 Sales Trend Over the Years")
    yearly_sales = df.groupby('year_of_release')['Units Sold (million )'].sum()
    st.line_chart(yearly_sales)
    
    # Bar Chart - Sales by Manufacturer
    st.subheader("🏭 Total Sales by Manufacturer")
    manufacturer_sales = df.groupby('manufacturer')['Units Sold (million )'].sum().sort_values(ascending=False)
    st.bar_chart(manufacturer_sales)
    
    # Pie Chart - Smartphones vs Non-Smartphones
    st.subheader("📱 Proportion of Smartphones vs Non-Smartphones")
    smartphone_counts = df['is_smartphone'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(smartphone_counts, labels=smartphone_counts.index, autopct='%1.1f%%', colors=['skyblue', 'orange'])
    st.pyplot(fig)
    
    # Stacked Bar Chart - Form Factor by Manufacturer
    st.subheader("📦 Sales by Form Factor")
    df_pivot = df.pivot_table(values='Units Sold (million )', index='manufacturer', columns='form_factor', aggfunc='sum', fill_value=0)
    df_pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')
    st.pyplot(plt)
    
    # Table - Top 10 Models
    st.subheader("📌 Top 10 Best-Selling Mobile Models")
    top_10_models = df[['model_name', 'manufacturer', 'year_of_release', 'Units Sold (million )']].nlargest(10, 'Units Sold (million )')
    st.dataframe(top_10_models)
