import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="wide")

st.title("📚 AI Book Recommendation System")

# -----------------------------
# Load Data (Robust + Smart)
# -----------------------------

@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv")

    # Clean columns
    books.columns = books.columns.str.strip().str.lower()

    # Rename to standard format
    books.rename(columns={
        'isbn': 'book_id',
        'book-title': 'title',
        'book-author': 'author',
        'image-url-m': 'image_url'
    }, inplace=True)

    books = books[['book_id', 'title', 'author', 'image_url']]
    books = books.drop_duplicates('title')

    # Reduce size for speed
    books = books.head(10000)

    # -----------------------------
    # Create realistic ratings
    # -----------------------------
    if not os.path.exists("ratings.csv"):

        data = []

        for user in range(1, 101):
            liked_books = np.random.choice(books['book_id'], 20)

            for book in liked_books:
                data.append([user, book, np.random.randint(3, 6)])

            random_books = np.random.choice(books['book_id'], 10)
            for book in random_books:
                data.append([user, book, np.random.randint(1, 3)])

        ratings = pd.DataFrame(data, columns=["user_id", "book_id", "rating"])
        ratings.to_csv("ratings.csv", index=False)

    else:
        ratings = pd.read_csv("ratings.csv")

    return books, ratings


books, ratings = load_data()

# -----------------------------
# Create Recommendation Matrix
# -----------------------------

@st.cache_data
def create_matrix(books, ratings):

    book_ratings = ratings.merge(books, on="book_id")
    book_ratings = book_ratings.drop_duplicates(subset=["user_id", "title"])

    user_book_matrix = book_ratings.pivot_table(
        index="title",
        columns="user_id",
        values="rating"
    ).fillna(0)

    # Safe filtering
    book_counts = book_ratings.groupby('title')['rating'].count()
    popular_books = book_counts[book_counts > 2].index

    if len(popular_books) == 0:
        popular_books = book_counts.index

    user_book_matrix = user_book_matrix.loc[
        user_book_matrix.index.intersection(popular_books)
    ]

    # Final safety
    if user_book_matrix.shape[0] == 0:
        return pd.DataFrame()

    similarity = cosine_similarity(user_book_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=user_book_matrix.index,
        columns=user_book_matrix.index
    )

    return similarity_df


with st.spinner("🔄 Building recommendation engine..."):
    similarity_df = create_matrix(books, ratings)

# -----------------------------
# Recommendation Function
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def content_based_recommendation(books):
    books['features'] = books['title'] + " " + books['author']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['features'])

    similarity = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=books['title'],
        columns=books['title']
    )

    return similarity_df
content_sim_df = content_based_recommendation(books)

def recommend(book_name):

    # 1️⃣ Try collaborative filtering
    if not similarity_df.empty and book_name in similarity_df.index:
        similar_scores = similarity_df[book_name].sort_values(ascending=False)[1:6]
        if len(similar_scores) > 0:
            return similar_scores.index.tolist()

    # 2️⃣ Fallback → content-based (SMART FIX)
    if book_name in content_sim_df.index:
        similar_scores = content_sim_df[book_name].sort_values(ascending=False)[1:6]
        return similar_scores.index.tolist()

    # 3️⃣ Final fallback → random
    return books['title'].sample(5).values
# -----------------------------
# UI - Book Selection
# -----------------------------

book_list = sorted(books['title'].dropna().unique())

selected_book = st.selectbox(
    "🔎 Search a book",
    book_list
)

# -----------------------------
# Show Recommendations
# -----------------------------

if st.button("Recommend"):

    recommended_books = recommend(selected_book)

    st.write("Recommended books:", recommended_books)  # remove later

    if not recommended_books:
        st.warning("⚠️ No strong matches found → showing popular books instead")

        recommended_books = books['title'].sample(5).values

    st.subheader("📖 Recommended Books")

    cols = st.columns(5)

    num_cols = 5

    for i in range(0, len(recommended_books), num_cols):
        cols = st.columns(num_cols)

        for j in range(num_cols):
            if i + j < len(recommended_books):

                book = recommended_books[i + j]
                book_data = books[books['title'] == book].iloc[0]

                with cols[j]:

                    if 'image_url' in book_data and pd.notna(book_data['image_url']):
                        st.image(book_data['image_url'], width=140)
                    else:
                        st.write("📘 No Image")

                    st.write(book)
                    st.caption(book_data['author'])
