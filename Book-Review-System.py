import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="wide")
st.title("📚 AI Book Recommendation System")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("Books.csv")

    books.columns = (
        books.columns
        .str.strip()
        .str.lower()
        .str.replace('-', '_')
        .str.replace(' ', '_')
    )

    def find_col(names):
        for n in names:
            if n in books.columns:
                return n
        return None

    book_id = find_col(['isbn', 'book_id'])
    title = find_col(['book_title', 'title'])
    author = find_col(['book_author', 'author'])
    image = find_col(['image_url_m', 'image_url_l', 'image_url'])

    if not book_id or not title or not author:
        st.error(f"Columns found: {books.columns.tolist()}")
        st.stop()

    df = pd.DataFrame({
        "book_id": books[book_id],
        "title": books[title],
        "author": books[author],
        "image_url": books[image] if image else None
    })

    df = df.dropna(subset=['title']).drop_duplicates('title').head(10000)

    return df


books = load_data()

# -----------------------------
# Content-based Recommendation
# -----------------------------
@st.cache_data
def build_model(df):
    df_copy = df.copy()
    df_copy["features"] = df_copy["title"] + " " + df_copy["author"]

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df_copy["features"])

    similarity = cosine_similarity(matrix)

    return pd.DataFrame(similarity, index=df_copy["title"], columns=df_copy["title"])


similarity_df = build_model(books)

def recommend(book):
    if book in similarity_df.index:
        return similarity_df[book].sort_values(ascending=False)[1:6].index.tolist()
    return books['title'].sample(5).tolist()


# -----------------------------
# UI
# -----------------------------
book_list = sorted(books['title'].unique())
selected = st.selectbox("🔎 Search a book", book_list)

if st.button("Recommend"):
    recs = recommend(selected)

    st.subheader("📖 Recommended Books")

    cols = st.columns(5)

    for i, book in enumerate(recs):
        data = books[books['title'] == book].iloc[0]

        with cols[i % 5]:
            if pd.notna(data['image_url']):
                st.image(data['image_url'], width=120)
            else:
                st.write("📘 No Image")

            st.write(book)
            st.caption(data['author'])
