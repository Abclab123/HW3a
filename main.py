import streamlit as st
import os
import tempfile
import camelot
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class PdfTableExtractor:
    def extract_tables(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages='all')
        return [table.df for table in tables]

class pdf2text:
    def __init__(self):
      pass

    def __call__(self, pdf_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            tables = camelot.read_pdf(tmpfile.name, pages="all")
            texts = []
        for table in tables:
            texts.append(table.df.to_string())
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def vectorize(self, text):
        # vector = self.model.encode(text)[0]
        return self.model.encode(text)[0]

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
      #2D arrary
        vector_from_table = np.array(vector_from_table).reshape(1, -1)
        vector_from_keyword = np.array(vector_from_keyword).reshape(1, -1)

        cosine_similar = cosine_similarity(vector_from_table, vector_from_keyword)
        return cosine_similar[0][0]

class PdfTableExtractor:
    def get_table(self, uploaded_file):
        # 使用 tempfile 创建一个临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            # 将 UploadedFile 对象的内容写入临时文件
            tmpfile.write(uploaded_file.getvalue())
            tmpfile_path = tmpfile.name

        # 使用临时文件的路径读取 PDF 文件
        pdf_result = camelot.read_pdf(tmpfile_path, pages="all")
        tables = []
        for table in pdf_result:
            tables.append(table.df)
        return tables

# Streamlit UI
def UI(keyword, uploaded_file):
    st.title("PDF Table Search")
    st.header("BD HW3a R12722047")

    if uploaded_file is not None:
        extractor = PdfTableExtractor()
        vectorizer = text2vector()
        similarity_calculator = cosine_sim()

        tables = extractor.get_table(uploaded_file)  # 获取 DataFrame 列表
        combined_text = "\n".join([table.to_string() for table in tables])  # 组合所有表格的文本
        table_vector = vectorizer.vectorize(combined_text)
        keyword_vector = vectorizer.vectorize(keyword)
        similarity_score = similarity_calculator(table_vector, keyword_vector)

        st.write(f"Similarity with the keyword: {similarity_score}")

        for table in tables:
            st.dataframe(table)


def main():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    keyword = st.text_input("Please enter a keyword")

    if st.button("Search") and uploaded_file and keyword:
        UI(keyword, uploaded_file)


if __name__ == "__main__":
    # main("keyword", "1.pdf")
    main()
