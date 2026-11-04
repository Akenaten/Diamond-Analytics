import streamlit as st
import plotly.express as px

from Utils.data_loader import load_data

st.set_page_config(
    page_title="Diamond Analytics",
    page_icon="💎",
    layout="wide"
)

df = load_data()

st.title("💎 Diamond Analytics Dashboard")
st.markdown(
    "Explore diamond prices, quality, "
    "and physical characteristics."
)

# Sidebar filters
st.sidebar.header("Filters")

selected_cuts = st.sidebar.multiselect(
    "Select Cut",
    options=sorted(df["cut"].unique()),
    default=sorted(df["cut"].unique())
)

filtered_df = df[df["cut"].isin(selected_cuts)]

if filtered_df.empty:
    st.warning("No diamonds match the selected filters.")
    st.stop()

# KPI metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Diamonds",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average Price",
    f"${filtered_df['price'].mean():,.2f}"
)

col3.metric(
    "Average Carat",
    f"{filtered_df['carat'].mean():.2f}"
)

col4.metric(
    "Median Price",
    f"${filtered_df['price'].median():,.2f}"
)

# Charts
left, right = st.columns(2)

with left:
    fig = px.histogram(
        filtered_df,
        x="price",
        nbins=50,
        title="Diamond Price Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.scatter(
        filtered_df,
        x="carat",
        y="price",
        color="cut",
        title="Carat vs Price",
        opacity=0.5
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    
    
    
    
    
    # import streamlit as st
# import pandas as pd
# import pathlib

# dpath = pathlib.Path.joinpath(pathlib.Path.cwd(), "Data", "diamonds.csv")
# file = pd.read_csv(dpath)
# file.info()
# # file.
# # title = st.title("Diamond Data Analysis")
# # st.write("This application analyzes data from the diamond dataset.")
