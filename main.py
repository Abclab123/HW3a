import os
import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st


class PDF2Text:
    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = [table.df.to_string().replace("\\n", "") for table in tables]
        return texts, tables


class Text2Vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text)


class CosineSimilarity:
    def __call__(self, vector_from_table, vector_from_keyword):
        vec1 = np.array(vector_from_table)
        vec2 = np.array(vector_from_keyword)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_most_relevant_table(keywords, pdf_folder):
    pdf_parser = PDF2Text()
    cosine = CosineSimilarity()
    encoder = Text2Vector()

    docs_folder = os.path.join(os.path.dirname(__file__), pdf_folder)

    best_table = None
    max_similarity = -1

    for pdf_file in os.listdir(docs_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(docs_folder, pdf_file)

            table_text, tables = pdf_parser(pdf_path)

            keywords_vec = encoder(keywords)

            for i, text in enumerate(table_text):
                similarity = cosine(encoder(text), keywords_vec)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_table = tables[i].df

    return best_table


if __name__ == "__main__":
    st.title("BDS HW3a")
    pdf_folder = "docs"  # Default to "docs" folder
    st.write("All PDF files under the 'docs' folder will be searched")
    st.subheader("Input Keyword")
    keyword = st.text_input("Keyword")
    submit_button = st.button("Search")

    if submit_button and keyword:
        with st.spinner('Searching'):
            best_table = find_most_relevant_table(keyword, pdf_folder)

            if best_table is not None:
                st.subheader("Most Relevant Table")
                st.write(best_table)
            else:
                st.subheader("No matching table found.")
