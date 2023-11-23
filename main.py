import camelot
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from numpy.linalg import norm
import streamlit as st
import streamlit.components.v1 as components


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        dfs = []
        texts = []
        for table in tables:
            content = table.df.to_string().replace("\\n", "")
            dfs.append(table.df.replace('\n', '', regex=True))
            texts.append(content)
        return dfs, texts


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer(
            'sentence-transformers/distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        embeddings = self.model.encode(text)
        return embeddings


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine = np.dot(vector_from_table, vector_from_keyword) / \
            (norm(vector_from_table)*norm(vector_from_keyword))
        return round(cosine, 4)


@st.cache_data
def load_file_list(folder):
    file_list = []
    for path in os.listdir(folder):
        file_path = os.path.join(folder, path)
        file_list.append(file_path)
    return file_list


def main(pdf_folder):
    st.title("Search PDF Table")

    file_list = load_file_list(pdf_folder)

    selected_files = st.multiselect('Choose files', file_list)
    keyword = st.text_input("Enter a keyword:")
    if selected_files and keyword and keyword != "":
        with st.spinner('Searching...'):
            pdf_parser = pdf2text()
            text_transformer = text2vector()
            sim_calculator = cosine_sim()
            table_texts = []
            table_dfs = []
            vectors = []
            for file_path in selected_files:
                dfs, texts = pdf_parser(file_path)
                for i in range(len(texts)):
                    embedding = text_transformer(texts[i])
                    table_texts.append(texts[i])
                    table_dfs.append(dfs[i])
                    vectors.append(embedding)

            keyword_vector = text_transformer(keyword)
            max_score = 0
            candidate = None
            for i in range(len(table_texts)):
                score = sim_calculator(vectors[i], keyword_vector)
                if score > max_score:
                    max_score = score
                    candidate = table_dfs[i]

            print(candidate)

        try:
            new_header = candidate.iloc[0]
            candidate = candidate[1:]
            candidate.columns = new_header
            st.dataframe(candidate, hide_index=True, column_config=None)
        except:
            st.write("No result.")


if __name__ == "__main__":
    main("docs/")
