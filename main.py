#R11631024 BDS Hw3 
#109

# 

import camelot
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PDF2Text:
    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")

        table_data = []

        for table in tables:
            table.df.replace('\n', '', regex=True, inplace=True)
            table_data.append({
                "df": table.df,
                "string": table.df.to_string(),
            })

        return table_data


class Text2Vector:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, text):
        return self.model.encode(text)


class CosineSimilarity:
    def __call__(self, vector_from_table, vector_from_keyword):
        return util.cos_sim(vector_from_table, vector_from_keyword)


class TableFinder:
    def __init__(self):
        self.pdf_parser = PDF2Text()
        self.encoder = Text2Vector()
        self.compute_sim = CosineSimilarity()

    def find_table(self, keyword, pdf_files):
        keyword_embed = self.encoder(keyword)

        best_sim = 0
        best_table_df = None

        for pdf_file in pdf_files:
            table_data = self.pdf_parser(pdf_file)

            for entry in table_data:
                table_embed = self.encoder(entry["string"])
                sim = self.compute_sim(table_embed, keyword_embed)
                if sim > best_sim:
                    best_sim = sim
                    best_table_df = entry["df"]

        return best_table_df


class UI:
    def __init__(self):
        self.table_finder = TableFinder()

    def show(self):
        st.set_page_config(page_title="PDF table")
        st.title("PDF table")

        pdf_info = {
            "1.pdf": "Deep learning related tables",
            "2.pdf": "Animal and plants",
        }

        for pdf, intro in pdf_info.items():
            st.write(f"{pdf}: {intro}")

        pdf_files = ["docs/1.pdf", "docs/2.pdf"]
        selected_pdf = st.multiselect("Select PDF files", pdf_files)

        keyword = st.text_input("Enter a keyword")

        if selected_pdf and keyword:
            with st.spinner("Searching.."):
                table = self.table_finder.find_table(keyword, selected_pdf)
            st.header("Search Result")
            st.table(table)


if __name__ == "__main__":
    ui = UI()
    ui.show()
