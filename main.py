import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        with open('docs/temp.pdf', "wb") as f:
            f.write(pdf_file.getvalue())
        tables = camelot.read_pdf('docs/temp.pdf', pages="all")

        #tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            text = table.data
            text = [' '.join(row) for row in text]
            text = ' '.join(text)
            text = text.replace('\n', '')
            texts.append(text)
        return texts

class text2vector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def __call__(self, texts):
        return self.vectorizer.fit_transform(texts)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cosine_similarity(vector_from_table, vector_from_keyword)

def main(keyword, pdf_files):
    pdf_parser = pdf2text()

    vector_parser = text2vector()
    cosine_parser = cosine_sim()
    #table_vectors = vector_parser(texts + [keyword])
    max_sim = -1
    max_sim_df = None
    for pdf_file in enumerate(pdf_files):
        table_text = pdf_parser(pdf_file[1])
        table_vectors = vector_parser(table_text + [keyword])
        #keyword_vec = vector_parser(keyword)
        for i in range( 0, table_vectors.shape[0] - 1 ):
            cand = cosine_parser(table_vectors[i], table_vectors[-1])
            if cand > max_sim:
                max_sim = cand
                with open('docs/temp.pdf', "wb") as f:
                    f.write(pdf_file[1].getvalue())
                tables = camelot.read_pdf('docs/temp.pdf', pages="all")
                max_sim_df = tables[ i ].df
    return max_sim_df

if __name__ == "__main__":

    st.title("BDS HW3a")
    st.write("R11922A17 Hong-Siou Chen")
    st.write("Develop a program capable of scanning multiple PDF files to identify the specific table containing the desired information.")

    pdfs = st.file_uploader("請上傳僅包含表格內容的PDF檔案。", type=['pdf'], accept_multiple_files=True)
    keyword = st.text_input("Searching keywords")

    if st.button("Submit", type="primary"):
        if pdfs == []:
            st.error("PDF files are required.")
            exit()
        elif keyword == "":
            st.error("Searching keywords are required.")
            exit()
        else:
            with st.spinner('Processing...'):
                table = main(keyword, pdfs)
                st.write("Result table:")
                st.write(table)