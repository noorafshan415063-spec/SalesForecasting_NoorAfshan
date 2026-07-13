import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide"
)

st.title("Retail Sales Forecasting Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )
    return df

df = load_data()

st.write(df.head())


# Sidebar

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

if page == "Sales Overview":

    st.header("Sales Overview Dashboard")

    total_sales = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    total_categories = df["Category"].nunique()

    total_regions = df["Region"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"${total_sales:,.0f}")

    c2.metric("Total Orders", total_orders)

    c3.metric("Categories", total_categories)

    c4.metric("Regions", total_regions)

    
    # Sales by Year
 
    st.subheader("Total Sales by Year")

    yearly_sales = (
        df.groupby(df["Order Date"].dt.year)["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        yearly_sales["Order Date"].astype(str),
        yearly_sales["Sales"]
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    ax.set_title("Total Sales by Year")

    st.pyplot(fig)


    # Monthly Sales Trend
    
    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"]
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Monthly Sales Trend")

    st.pyplot(fig)


       
    # Interactive Filters
  
    st.subheader("Sales by Region and Category")

    region = st.selectbox(
        "Select Region",
        ["All"] + sorted(df["Region"].unique().tolist())
    )

    category = st.selectbox(
        "Select Category",
        ["All"] + sorted(df["Category"].unique().tolist())
    )

    filtered_df = df.copy()

    if region != "All":
        filtered_df = filtered_df[
            filtered_df["Region"] == region
        ]

    if category != "All":
        filtered_df = filtered_df[
            filtered_df["Category"] == category
        ]

    st.dataframe(
        filtered_df[
            ["Order Date","Region","Category","Sales"]
        ]
    )


   
# Forecast Explorer


elif page == "Forecast Explorer":

    st.header("Forecast Explorer")

    forecast_option = st.selectbox(
        "Select Category / Region",
        [
            "Furniture",
            "Technology",
            "Office Supplies",
            "West Region",
            "East Region"
        ]
    )

    months = st.slider(
        "Forecast Horizon (Months)",
        min_value=1,
        max_value=3,
        value=3
    )

    forecast_data = {
        "Furniture": {
            "Forecast": [10143.45, 5940.99, 17309.04],
            "MAE": 2732.74,
            "RMSE": 2989.82,
            "MAPE": 8.86
        },
        "Technology": {
            "Forecast": [16156.24, 17111.81, 21681.73],
            "MAE": 10745.89,
            "RMSE": 14195.68,
            "MAPE": 25.14
        },
        "Office Supplies": {
            "Forecast": [22217.07, 22210.15, 23817.33],
            "MAE": 3976.28,
            "RMSE": 4048.76,
            "MAPE": 14.47
        },
        "West Region": {
            "Forecast": [14060.43, 14976.06, 26887.54],
            "MAE": 4455.01,
            "RMSE": 5942.04,
            "MAPE": 16.92
        },
        "East Region": {
            "Forecast": [16933.03, 19684.63, 15576.39],
            "MAE": 12472.70,
            "RMSE": 14919.33,
            "MAPE": 33.97
        }
    }

    months_name = [
        "January 2019",
        "February 2019",
        "March 2019"
    ]

    forecast_df = pd.DataFrame({
        "Forecast Month": months_name[:months],
        "Forecast Sales":
            forecast_data[forecast_option]["Forecast"][:months]
    })

    st.subheader("Forecast Output")

    st.dataframe(forecast_df)

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        forecast_df["Forecast Month"],
        forecast_df["Forecast Sales"],
        marker="o"
    )

    ax.set_xlabel("Forecast Month")
    ax.set_ylabel("Forecast Sales")
    ax.set_title(f"{forecast_option} Forecast")

    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f'{forecast_data[forecast_option]["MAE"]:.2f}'
    )

    c2.metric(
        "RMSE",
        f'{forecast_data[forecast_option]["RMSE"]:.2f}'
    )

    c3.metric(
        "MAPE",
        f'{forecast_data[forecast_option]["MAPE"]:.2f}%'
    )


    
# Anomaly Report


elif page == "Anomaly Report":

    st.header("Anomaly Detection Report")

    st.write("""
    This page presents unusual sales patterns detected during the analysis.
    Two anomaly detection techniques were applied:
    
    • Isolation Forest (Machine Learning Based)
    • Z-Score Method (Statistical Method)

    These methods help identify unexpected sales spikes and drops that may be
    caused by promotional campaigns, seasonal demand, inventory issues, or
    unusual business events.
    """)

   
    # Isolation Forest
   

    st.subheader("Isolation Forest Anomaly Detection")

    st.image(
    "charts/isolation_forest_anomaly.png",
    caption="Isolation Forest detected anomalies in weekly sales.",
    use_container_width=True
)

    
    # Z-Score
  

    st.image(
    "charts/zscore_anomaly.png",
    caption="Anomalies detected using the Z-Score statistical method.",
    use_container_width=True
)
   
    # Business Summary
    

    st.subheader("Business Summary")

    st.success("""
    The detected anomalies indicate weeks where sales were significantly
    different from the normal sales pattern.

    These unusual observations may represent:

    • Seasonal demand
    • Festival sales
    • Promotional campaigns
    • Inventory shortages
    • Unexpected market behaviour

    Monitoring these anomalies helps businesses improve inventory planning
    and avoid stock shortages or overstock situations.
    """)


    
# Product Demand Segments


elif page == "Product Demand Segments":

    st.header("Product Demand Segmentation")

    st.write("""
    Products were grouped into different demand segments using the K-Means clustering algorithm.
    The clustering was performed using four business features:

    • Total Sales
    • Average Order Value
    • Sales Volatility
    • Sales Growth

    This helps identify products with similar demand behaviour, allowing businesses
    to make better stocking and inventory planning decisions.
    """)


    # Cluster Chart
  

    st.subheader("Product Demand Clusters")

    st.image(
        "charts/product_clusters.png",
        caption="K-Means clustering of product demand behaviour.",
        use_container_width=True
    )

    
    # Cluster Table
   

    st.subheader("Sub-Categories and Assigned Clusters")

    cluster_table = pd.DataFrame({

        "Sub-Category":[
            "Accessories",
            "Appliances",
            "Art",
            "Binders",
            "Bookcases",
            "Chairs",
            "Copiers",
            "Envelopes",
            "Fasteners",
            "Furnishings",
            "Labels",
            "Machines",
            "Paper",
            "Phones",
            "Storage",
            "Supplies",
            "Tables"
        ],

        "Cluster":[
            0,1,1,0,0,0,2,1,1,1,1,2,1,0,0,1,1
        ]

    })

    st.dataframe(cluster_table)

    
    # Cluster Statistics
   

    st.subheader("Cluster Summary")

    cluster_summary = pd.DataFrame({

        "Cluster":[0,1,2],

        "Total Sales":[224662.88,64230.31,167743.36],

        "Average Order Value":[337.42,158.56,1930.72],

        "Sales Volatility":[491.29,263.82,2990.64],

        "Sales Growth":[3.39,9.94,0.14]

    })

    st.dataframe(cluster_summary)

    
    # Business Recommendations
   

    st.subheader("Recommended Stocking Strategy")

    st.success("""

Cluster 0
• Maintain regular inventory levels.
• Products show stable demand and consistent sales.

Cluster 1
• Keep moderate inventory.
• Monitor demand regularly because sales growth is comparatively higher.

Cluster 2
• Maintain safety stock.
• These products have very high order values and high sales volatility,
  so inventory should be carefully monitored to avoid stock shortages.

""")
