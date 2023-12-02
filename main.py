import camelot
import numpy as np
import pandas as pd
from numpy.linalg import norm
import streamlit as st
from sentence_transformers import SentenceTransformer

class TableExtractor:
    def __init__(self):
        self.tables_data = []

    def extract_tables(self, pdf_file_path: str):
        # Use Camelot to parse tables in the PDF
        tables = camelot.read_pdf(pdf_file_path, pages="all")
        print("Total tables extracted:", tables.n)
        for table in tables:
            # Convert table data to a dictionary, including DataFrame and text representation of the table
            self.tables_data.append({
                "df": table.df,
                "text": table.df.to_string().replace("\\n", "")
            })
        return self.tables_data

class Vectorizer:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def encode(self, text):
        # Use the model to convert text to vectors
        return self.model.encode(text, convert_to_tensor=False)

class CosineSimilarityCalculator:
    def __call__(self, vector_from_table, vector_from_keyword):
        # Calculate cosine similarity between two vectors
        cosine_similarity = np.dot(vector_from_table, vector_from_keyword) / (norm(vector_from_table) * norm(vector_from_keyword))
        return cosine_similarity

def search_best_table(keyword, pdf_file_path) -> pd.DataFrame:
    table_extractor = TableExtractor()
    # Extract tables from the PDF
    tables_data = table_extractor.extract_tables(pdf_file_path)
    
    vectorizer = Vectorizer()
    # Convert table text to vectors
    vectors = [vectorizer.encode(table["text"]) for table in tables_data]
    
    cosine_similarity_calculator = CosineSimilarityCalculator()
    # Calculate cosine similarity
    cosine_scores = [cosine_similarity_calculator(vector, vectorizer.encode(keyword)) for vector in vectors]

    return tables_data[np.argmax(cosine_scores)]["df"]

def UI():
    st.title('PDF Table Search Engine')
    st.write("Please upload your PDF file")
    upload_file = st.file_uploader("Upload YOUR PDF File", type=['pdf'])

    if upload_file is not None:
        st.write("You uploaded the file")
        st.write(upload_file.name)
        pdf_file = f"docs/{upload_file.name}"
        with open(pdf_file, 'wb') as f:
            f.write(upload_file.read())

    st.write("Please enter the keyword you want to search for")
    keyword = st.text_input("Keyword", "Application of Unsupervised Learning")

    if st.button("Search"):
        # Call the search function and display the result
        table = search_best_table(keyword, pdf_file)
        st.write("Result")
        st.dataframe(table)

if __name__ == "__main__":
    UI()
