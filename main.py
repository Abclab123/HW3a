import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import streamlit as st
import io
import tempfile
import os
class pdf2text:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def extract_text(self):
        tables = camelot.read_pdf(self.pdf_file, pages="all")
        extracted_tables = []
        for table in tables:
            table_dict = {
                'df': table.df,
                'text': {'text': table.df.to_string().replace("\\n", "")}
            }
            extracted_tables.append(table_dict)
        return extracted_tables


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        similarity = np.dot(vector_from_table, vector_from_keyword) / (norm(vector_from_table) * norm(vector_from_keyword))
        return similarity


def searchTable(keyword, uploaded_file):
    # Initialization
    cos_sim = cosine_sim()

    # 將上傳的 PDF 內容保存到臨時文件中
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    try:
        # 使用臨時文件的路徑進行解析
        pdf_parser = pdf2text(temp_pdf_path)
        vectorizer = text2vector()

        # Get the tables
        tables = pdf_parser.extract_text()  # 使用 extract_text 方法

        # 檢查文件是否為空
        if not tables:
            print("The PDF file is empty.")
            return None

        # Get the similarity scores
        sim = [cos_sim(vectorizer(t['text']['text']), vectorizer(keyword)) for t in tables]

        # Output
        output_table = tables[np.argmax(sim)]
        print("Table: ")
        print(output_table['df'])
        return output_table
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        # 刪除臨時文件
        os.remove(temp_pdf_path)


def main():
    st.set_page_config(page_title='AI Table')
    st.title(':cherry_blossom: :rainbow[AI Table]')

    # 讓使用者上傳 PDF 文件
    uploaded_file = st.file_uploader("Upload a PDF")

    if uploaded_file is not None:
        # 如果有上傳文件，顯示搜索按鈕
        if st.button("Search"):
            # 讓使用者輸入關鍵字
            keyword = st.text_input("Enter Keyword")

            # 使用上傳的文件進行搜索
            search_result = searchTable(keyword, uploaded_file)

            # 顯示搜索結果
            if search_result is not None:
                st.write("Table:")
                st.dataframe(search_result['df'])

if __name__ == "__main__":
    main()



