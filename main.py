import camelot
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import streamlit as st
import tempfile


class pdf2text:
    def __call__(self, pdf_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.read())
            tmpfile_path = tmpfile.name
        tables = camelot.read_pdf(tmpfile_path, pages="all")
        texts = []
        for table in tables:
            text = table.df.to_string()
            text = text.replace('\n', '')
            texts.append({
                "df": table.df,
                "string": text,
            })
        return texts  # Return list of texts for each table

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("thenlper/gte-large")

    def __call__(self, texts):
        return self.model.encode(texts)

class cosine_sim:
    def __call__(self, vector_from_table, vector_from_keyword):
        return util.cos_sim(vector_from_table, vector_from_keyword)


def analyze_pdf(keyword, pdf_file):
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file)
    
    vectorizer = text2vector()
    
    tables_text = [i["string"] for i in tables]
    
    tables_vector = vectorizer(tables_text)
    keyword_vector = vectorizer(keyword)
    test_sim = cosine_sim()
    similarity = test_sim(keyword_vector, tables_vector)
    highest = np.argmax(similarity, axis=-1)
    
    return tables[highest[0].item()]["df"]

# Streamlit interface
st.title("PDF Table Search Tool")

# User input for the search keyword
keyword = st.text_input("Enter the search keyword:")

# File uploader allows user to add their own PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# When the user uploads a file and enters a keyword, process the PDF
if uploaded_file is not None and keyword:
    with st.spinner('Analyzing...'):
        result = analyze_pdf(keyword, uploaded_file)
        st.write(result)
