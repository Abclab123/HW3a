import threading
import camelot
import os
from glob import glob
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util


scores = []
class Table:
    def __init__(self, dataframe, text, title = "unamed") -> None:
        self.title = title
        self.dataframe = dataframe
        self.frametext = text

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(Table(table.df, table.df.to_string().replace("\\n", "")))
        return texts 


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("thenlper/gte-base-zh")

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword).item()


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_frames = pdf_parser(pdf_file)

    t2v = text2vector()
    cos_sim = cosine_sim()

    keyword = t2v(keyword)

    max_score = -1
    ans = ""

    for frame in table_frames:
        vec = t2v(frame.frametext)
        
        score = cos_sim(vec, keyword)
        if score > max_score:
            max_score = score
            ans = frame
    return ans, max_score

if __name__ == "__main__":
    st.title("HW3a - Big Data Systems B08902093")

    uploaded_files = st.file_uploader("Upload pdf files", accept_multiple_files=True, type=['pdf'])
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            with open(os.path.join("docs", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File " + uploaded_file.name + "uploaded")
        
    files = [f for f in glob("docs/*.pdf")]
    selected = st.multiselect("Select the pdf files to be searched", files, default=files)
    keyword = st.text_input("Please enter your keyword: ")

    if keyword and selected:
        with st.spinner("Searching"):
            ans = Table(dataframe=None, text="")  
            max_score = -1
            file = ""
            for f in selected:
                table_data, score = main(keyword, f)
                if score > max_score:
                    max_score = score
                    ans = table_data
                    file = f
                print(score, f)
        st.subheader("From table " + file)
        st.dataframe(ans.dataframe)
