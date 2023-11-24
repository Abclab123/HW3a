import camelot
from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy.linalg import norm
import streamlit as st
import os


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        table_df = []
        for table in tables:
            table_df.append(table.df)
            text = table.df.to_string()
            text = text.replace("\\n", "")
            texts.append(text)
        return texts, table_df



class text2vector:
    def __init__(self):
        # pass
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        # pass
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine = np.dot(vector_from_table,vector_from_keyword)/(norm(vector_from_table)*norm(vector_from_keyword))
        return cosine

def UI():
    pass
    

def main(keyword, pdf_file):
    # parser
    pdf_parser = pdf2text()
    text2vector_parser = text2vector()
    cosine_sim_parser = cosine_sim()
    # table to text
    table_text, table_df = pdf_parser(pdf_file)
    # text to vector
    table_vector = text2vector_parser(table_text)
    keyword_vector = text2vector_parser(keyword)
    
    # cosine similarity
    cos_sim = []
    for i in range(len(table_vector)):
        cos_sim.append(cosine_sim_parser(table_vector[i], keyword_vector))
    
    # find the max cosine similarity
    max_cos_sim = max(cos_sim)
    max_index = cos_sim.index(max_cos_sim)
    table = table_df[max_index]

    return table


if __name__ == "__main__":
    UI()