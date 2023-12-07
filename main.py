import camelot
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file): 
        with open('temp.pdf', "wb") as f:
            f.write(pdf_file.read())
        raw_tables = camelot.read_pdf('temp.pdf', pages="all")
        
        tables = {"text":[], "df":[]}
        for table in raw_tables:
            tables["text"].append(table.df.to_string().replace("\\n", ""))
            df = table.df.replace("\\n", "", regex=True)
            while df.iloc[0][0] == "":
                df = df.iloc[1:].reset_index(drop=True)
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
            df = df.set_index(df.columns[0])
            tables["df"].append(df)
            
        os.remove('temp.pdf')
        return tables


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.cos_sim(vector_from_table, vector_from_keyword)[0][0]


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text_parser = text2vector()
    cos_sim = cosine_sim()
    
    tables = pdf_parser(pdf_file)
    tables_vec = text_parser(tables["text"])
    keyword_vec = text_parser(keyword)

    cos_sim_scores = [cos_sim(table_vec, keyword_vec) for table_vec in tables_vec]   
    
    df = tables["df"][np.argmax(cos_sim_scores)]
    return df

def UI():
    st.title("BDS HW3a: PDF Table Search")
    
    st.subheader("Upload PDF file")
    pdf = st.file_uploader("Upload pdf file", type=['pdf'], label_visibility="hidden")
    
    st.subheader("Search keyword")
    keyword = st.text_input("Search keyword", label_visibility="hidden")
    
    search = st.button("Search")

    if search:
        if pdf == []:
            st.error("Need to upload PDF file.")
            return
        
        with st.spinner('Processing...'):
            st.subheader("Desired table:")
            table = main(keyword, pdf)
            st.write(table)

if __name__ == "__main__":
    UI()
