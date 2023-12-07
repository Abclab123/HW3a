import camelot
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import numpy as np
import streamlit as st
import os
import tempfile

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        streams = camelot.read_pdf(pdf_file, flavor='stream', pages='all')
        table_titles = []
        texts = []
        table_df = []
        for table in tables:
            table_df.append(table.df)
            text = table.df.to_string()
            text = text.replace("\\n", "")
            texts.append(text)
        for s in streams:
            s_texts=s.df.to_string().split('\n')
            for t in s_texts:
                if "ai_tables" in t:
                    table_titles.append(''.join(t.split()[1:]))
        return texts, table_df, table_titles

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return np.dot(vector_from_table, vector_from_keyword) / (norm(vector_from_table)*norm(vector_from_keyword))

def UI():
    st.title('BDS HW3a')
    pdf_file = st.file_uploader("PDF File:", type='pdf')
    keyword = st.text_input("Enter keyword:")

    if pdf_file is not None and keyword != '':
        # Read the content of the uploaded PDF into an in-memory buffer
        pdf_content = pdf_file.read()
        # Create a temporary file to save the uploaded PDF
        temp_pdf_file = tempfile.NamedTemporaryFile(dir='.', suffix='.pdf', delete=False)

        # Write the PDF content to the temporary file
        temp_pdf_file.write(pdf_content)

        # Close the temporary file to ensure it is properly saved
        temp_pdf_file.close()

        if st.button("Search"):
            title, table = main(keyword, temp_pdf_file.name)
            st.subheader("Result")
            st.write(title)
            st.write(table)
        os.remove(temp_pdf_file.name)

def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text2vec_parser = text2vector()
    cosine_sim_parser = cosine_sim()

    table_texts, table_df, titles = pdf_parser(pdf_file)
    text_vec = text2vec_parser(table_texts)
    key_vec = text2vec_parser(keyword)
    table_sim = []
    for v in text_vec:
        table_sim.append(cosine_sim_parser(v, key_vec))
    max_index = table_sim.index(max(table_sim))
    title = titles[max_index]
    table = table_df[max_index]
    return title, table


if __name__ == "__main__":
    UI()