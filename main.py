import camelot
import jieba
import streamlit as st
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages='all')

        texts = []
        for table in tables:
            data = table.data
            data = map(lambda l: ' '.join(l), data)
            data = ' '.join(data)
            data = data.replace('\n', '')
            texts.append(data)
        return texts

    def get_table(self, pdf_file, index):
        tables = camelot.read_pdf(pdf_file, pages='all')
        return tables[index].df

class text2vector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=lambda text: list(jieba.cut(text)))

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, table_vector, keyword_vector):
        return cosine_similarity(table_vector, keyword_vector)

def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_texts = pdf_parser(pdf_file)

    keyword_vectorizer = text2vector()
    table_vectorizer = text2vector()

    all_texts = [keyword] + table_texts  # Combine keyword and table texts
    all_vectors = keyword_vectorizer.fit_transform(all_texts)
    
    keyword_vector = all_vectors[0]
    table_vectors = all_vectors[1:]

    similarity_calculator = cosine_sim()
    similarity = similarity_calculator(table_vectors, keyword_vector)

    if similarity.size > 0:
        max_sim_index = similarity.argmax()
        if similarity[max_sim_index] > 0:
            max_sim_table = pdf_parser.get_table(pdf_file, max_sim_index)
            st.write("Most similar table:")
            st.table(max_sim_table)
        else:
            st.write("Keyword not found in the table.")
    else:
        st.write("No tables found in the document.")

if __name__ == "__main__":
    st.title("Welcome to the keyword search system of Black Cat🐈‍ (B11902014 高浩鈞)")
    input_file = st.file_uploader("Please upload your PDF file: ", type='pdf')
    
    if input_file:
        temp_dir = tempfile.mkdtemp(dir='.', suffix='.pdf')
        file_path = os.path.join(temp_dir, 'uploaded_file.pdf')
        with open(file_path, "wb") as f:
            f.write(input_file.getvalue())
    else:
        st.warning("Please upload a PDF file.")
        st.stop()
        
    keyword = st.text_input("Please enter a keyword: ")
    if not keyword:
        st.warning("Please enter a keyword.")
        st.stop()
    
    main(keyword, file_path)
