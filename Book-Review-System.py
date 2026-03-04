import streamlit as st
import pandas as pd
st.title("Book Review System")
st.subheader("By Saubhagya Munsi")

file = st.file_uploader("Upload your file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df)
if file:
    st.subheader("Data Summary")
    st.write(df.describe())
if file:
    author = df["Book-Author"].unique()
    selected_books = st.selectbox("Select Author", author)
    filtered_data = df[df["Book-Author"] == selected_books]
    st.dataframe(filtered_data)