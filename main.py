import camelot 
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        pdf_tables = camelot.read_pdf(pdf_file, pages="all")
        tables = []
        for table in pdf_tables:
            table.df = table.df[~table.df.map(lambda x: x.strip() == "").all(axis=1)]
            max_len = max([len(t) for t in table.df.iloc[0] if isinstance(t, str)])

            if max_len > 10:
                tables[-1] = pd.concat([tables[-1], table.df], axis=0, ignore_index=True)
            else:
                tables.append(table.df)

        return tables
    
    def getTitle(self, pdf_file, idx):
        pdf_tables = camelot.read_pdf(pdf_file, flavor="stream", pages="all")
        titles = []
        for table in pdf_tables:
            data = table.data
            data = [''.join(row) for row in data]
            for text in data:
                if 'ai_tables' in text:
                    titles.append(text)
        titles = np.unique(titles)
        return titles[idx]


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.cos_sim(vector_from_table, vector_from_keyword)


def search(keyword, pdf_file, op):
    if op==1:
        path = os.path.join('docs', pdf_file)
    elif op==2:
        path = os.path.join("docs", pdf_file.name)
        with open(path, "wb") as f:
            f.write(pdf_file.read())
    if path == "" or os.path.getsize(path) == 0:
        st.markdown('<span style="color: red;">Input error!</span>', unsafe_allow_html=True)
        return

    pdf_parser = pdf2text()
    txt2vec = text2vector()
    cos_sim = cosine_sim()
    
    tables = pdf_parser(path)
    vec_ky = txt2vec(keyword)
    eval = []

    for table in tables:
        text = table.to_string().replace("\n", "")
        text = text.replace("\\n", "")
        vec_table = txt2vec(text)
        eval.append(cos_sim(vec_table, vec_ky))

    max_idx = eval.index(max(eval))
    print(eval)
    st.subheader("Output")
    st.text(pdf_parser.getTitle(path, max_idx))
    st.dataframe(tables[max_idx])


if __name__ == "__main__":
    st.title("BDS HW3a - R12921A09")
    st.divider()
    st.subheader("Choose a PDF file")

    PDF_files = os.listdir("./docs")
    if st.selectbox("Choose", ("Choose an existing PDF", "Upload a PDF"), label_visibility="hidden")=="Choose an existing PDF":
        pdf = st.selectbox("Choose a PDF", PDF_files, label_visibility="hidden")       
        op = 1
    else:
        pdf = st.file_uploader("upload a PDF", type=['pdf'], label_visibility="hidden")
        op = 2

    st.subheader("Input a keyword")
    kw = st.text_input("input keyword", label_visibility="hidden")

    if st.button("Search"):
        if kw=="" or pdf == None:
            st.markdown('<span style="color: red;">Input error!</span>', unsafe_allow_html=True)
        else:
            search(kw, pdf, op)
