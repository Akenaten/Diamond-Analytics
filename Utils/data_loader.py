import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("data/diamonds.csv")
    #st.write("Dataset columns:", df.columns.tolist())
    st.write("Dataset preview:", df.head())



    # Remove optional index column
    df = df.loc[
        :, ~df.columns.str.startswith("Unnamed:")
    ]

    # Remove duplicate records
    # df = df.drop_duplicates()
    
    

    # # Exclude invalid physical dimensions
    # df = df[
    #     (df["x"] > 0) &
    #     (df["y"] > 0) &
    #     (df["z"] > 0)
    # ].copy()

    return df