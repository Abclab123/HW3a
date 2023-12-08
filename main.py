import tempfile

import camelot
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file_content):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_file.write(pdf_file_content)
            temp_file.seek(0)
            tables = camelot.read_pdf(temp_file.name, pages="all")
            return {table.df.to_string().replace('\\n', ''): table for table in tables}


class text2vec:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text_list):
        return self.model.encode(text_list)


def search_keyword_in_pdf(keyword, pdf_file_content):
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file_content)
    tables_text = list(tables.keys())

    text_parser = text2vec()
    tables_vec = text_parser(tables_text)
    keyword_vec = text_parser([keyword])

    cos_sim_list = cos_sim(tables_vec, keyword_vec)
    max_index = torch.argmax(cos_sim_list)  # Get the index of the max value

    return tables[tables_text[max_index]]


class UI:
    def __init__(self):
        st.title("PDF Table Search")
        self.pdf_file = st.file_uploader("Upload PDF", type="pdf")
        self.keyword = st.text_input("Keyword")
        self.search_button = st.button("Search")

        # the callback of search_button should call search_keyword_in_pdf
        if self.search_button:
            if self.pdf_file is None:
                st.write("Please upload a PDF file")
            elif self.keyword == "":
                st.write("Please input a keyword")
            else:
                pdf_file_content = self.pdf_file.read()
                table = search_keyword_in_pdf(self.keyword, pdf_file_content)
                st.write(table.df)


if __name__ == "__main__":
    ui = UI()

