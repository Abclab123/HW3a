import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
import os
import pandas as pd


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
	set_table_df = None
        temp_table_df = None
        
        tables_df = []
        texts = []
        
        for table in tables:
            if table.df.index[-1] <= 2:
                set_table_df = table.df
            else:
                if set_table_df is not None:
                    temp_table_df = pd.concat([set_table_df, table.df], ignore_index=True)
                    set_table_df = None
                else:
                    temp_table_df = table.df
            
            # collect each table
            tables_df.append(temp_table_df)
            # change table to text
            temp_text = [temp_table_df.to_string()]
            # collect texts in each table
            texts.append("\n".join(temp_text).replace("\\n", ""))
            
        return texts, tables_df


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        texts_vector = self.model.encode(text)
        return texts_vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        sim = np.dot(vector_from_table, vector_from_keyword.T) / (np.linalg.norm(vector_from_table) * np.linalg.norm(vector_from_keyword))
        return sim

def main(keyword, pdf_file):
    
    pdf_parser = pdf2text()
    table_text, table_df = pdf_parser(pdf_file)

    keyword_vectorizer = text2vector()
    keyword_vector = keyword_vectorizer(keyword)

    table_vectorizer = text2vector()
    table_vector = table_vectorizer(table_text)

    # calculate the similarity
    similarity_calculator = cosine_sim()
    similarity = []
    for i in range(len(table_df)):
        similarity.append(similarity_calculator(table_vector[i], keyword_vector))
    
    # find the index of the table with maximum similarity
    maximum_similarity = max(similarity)
    index_maximum_similarity = similarity.index(maximum_similarity)
    
    # return table
    return table_df[index_maximum_similarity]



if __name__ == "__main__":
    st.title('Big Data Systems HW3a 陳思妤')
    
    uploaded_file = st.file_uploader('請上傳一個 PDF 文件 (1 or 2)', type='pdf')
    st.write('註: 請上傳檔名為1(監督式學習、非監督式學習、強化學習)或2(動物、植物細胞)之 PDF 文件')
    if uploaded_file is not None:
        pdf_file = os.path.join("docs", uploaded_file.name)

    keyword = st.text_input('請輸入關鍵字', " ")

    if st.button('開始搜尋'):
        result_table = main(keyword, pdf_file)
        st.write('搜尋結果：')
        st.write(result_table)
