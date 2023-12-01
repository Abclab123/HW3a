import camelot
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
import streamlit as st
import pandas as pd

class StreamlitUI:
    
    def __init__(self, model):
        self.tmp_file = "tmp.pdf"
        self.model = model
        
    def start(self):
        st.title("BDS hw3a")
        st.header("B09902134 鄭鴻鈞")
        uploaded_file = self.pdf_uploader(self.tmp_file)
        if not uploaded_file:
            st.text("Please upload a pdf file")
            return
        pdf_parser = pdf2text()
        keyword = self.keyword_input()
        tables = pdf_parser.get_table_as_df(self.tmp_file)
        st.button("Search", on_click=lambda: self.start_search(keyword, tables))

        if "result_table" in st.session_state:
            st.table(st.session_state.result_table)

    def start_search(self, keyword, tables):
        st.session_state.result_table = self.find_table(tables, keyword)
        
    
    def find_table(self, tables: pd.DataFrame, keyword):
        return max(tables, key=lambda table: self.model.similarity(table.to_string(), keyword))

    def pdf_uploader(self, file_name, text = "Choose a PDF file" ):
        uploaded_file = st.file_uploader(text, type="pdf")
        
        if uploaded_file:
            if "file_name" in st.session_state and uploaded_file.name != st.session_state.file_name:
                del st.session_state.result_table
            st.session_state.file_name = uploaded_file.name
            self.write_file(file_name, uploaded_file.read())
        return uploaded_file
    
    def keyword_input(self):
        query = st.text_input("input keyword")
        return query

    
    def write_file(self, file_name, bytes):
        with open(file_name, "wb") as f:
            f.write(bytes)

class pdf2text:
    def __init__(self):
        pass

    def get_table_as_df(self, pdf_file):
        # pdf_file is value from st.file_uploader that only contains tables
        
        pdf_result = camelot.read_pdf(pdf_file, pages="all")
        tables = []
        for table in pdf_result:
            tables.append(table.df)
        return tables



class Model:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        

    def similarity(self, text1, text2):
        embedding = self.model.encode([text1, text2])
        return 1 - paired_cosine_distances([embedding[0]], [embedding[1]])
    


if __name__ == "__main__":
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = Model(model_name)
    StreamlitUI(model).start()
