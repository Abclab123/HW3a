import string
import tempfile

import camelot
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util

class pdf2text:

    strippers = string.digits + ' '

    def __init__(self):
        pass

    def table_to_text(self, table):
        keyword = table.df.to_string()
        for c in self.strippers:
            keyword = keyword.replace(c, '')
        return keyword.replace('\\n', '')

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        key_table_pair = {}
        return {self.table_to_text(table): table for table in tables}
        return key_table_pair


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=True)


def pdf_search(keyword, pdf_file):
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file)
    table_keywords = list(tables.keys())

    text_encoder = text2vector()
    keyword_vector = text_encoder([keyword])
    table_keyword_vectors = text_encoder(table_keywords)

    cos_scores = util.cos_sim(table_keyword_vectors, keyword_vector)
    most_relative_table_keyword = table_keywords[torch.argmax(cos_scores)]

    return tables[most_relative_table_keyword]


def main():
    st.title("Document Intelligence")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    keyword = st.text_input("Query")
    submit = st.button("Search")

    if submit:
        if pdf_file is None:
            st.write("Please upload a PDF file")
        elif keyword == '':
            st.write("Please input a keyword")
        else:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(pdf_file.read())
                table = pdf_search(keyword, tmp_pdf.name)
                st.write(table.df)


if __name__ == "__main__":
    main()
