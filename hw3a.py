#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import camelot
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, file):
        tmp_path = "tmp.pdf"
        with open(tmp_path, "wb") as f:
            f.write(file.getvalue())

        tables = camelot.read_pdf(tmp_path, pages="all")
        dfs = []
        for table in tables:
            dfs.append(table.df)

        Path.unlink(tmp_path)  # remove the tmp file

        return dfs


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword)

def main(keyword, files):
    pdf_parser = pdf2text()
    embedder = text2vector()
    get_sim = cosine_sim()

    keyword_embed = embedder(keyword)

    best_sim = 0
    best_table_df = None

    for i, file in enumerate(files):
        table_dfs = pdf_parser(file)

        for j, table_df in enumerate(table_dfs):
            table_text_embed = embedder(table_df.to_string())
            sim = get_sim(table_text_embed, keyword_embed)
            if sim > best_sim:
                best_sim = sim
                best_table_df = table_df

    return best_table_df

class ST:
    def ui(self):
        st.title("BDS Homework3 Stage-A")
        st.write("by R11246011 陳柏伍")

        file = st.file_uploader("Drop pdf with only tables inside.", accept_multiple_files=True)
        query = st.text_input("Type the searching keywords.")
        
        st.write("Press and see the result")
        if "alert" in st.session_state and st.session_state.alert != None:
            st.warning(st.session_state.alert)
        st.button("Search", type="primary", on_click=self.debug, args=(file, query))

        st.divider()

        if "table" in st.session_state:
            st.table(st.session_state.table)

    def debug(self, file, query):
        st.session_state["alert"] = None
        if file == []:
            st.session_state.alert = "files missing"
            return
        if query == "":
            st.session_state.alert = "keywards missing"
            return

        table_df = main(query, file)
        st.session_state.table = table_df

if __name__ == "__main__":
    ui = ST()
    ui.ui()

