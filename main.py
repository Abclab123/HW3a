import camelot
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            data = table.data
            data = [' '.join(row) for row in data]
            text = ' '.join(data)
            text = text.replace('\n', '')
            texts.append(text)
        return texts


class text2vector:
    def __init__(self):
        self.vectors = TfidfVectorizer()

    def __call__(self, text):
        return self.vectors.fit_transform(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_keyword, vector_from_table):
        return cosine_similarity(vector_from_keyword, vector_from_table)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text_vector = text2vector()
    similarity = cosine_sim()

    table_text = pdf_parser(pdf_file)

    keyword_split = jieba.lcut(keyword)
    table_vectors = text_vector(table_text + keyword_split)
    keyword_vectors = text_vector(keyword_split)
    
    print(table_vectors.shape)

    # for vector in table_vectors:
    #     results = []
    #     result = similarity(vector, table_vectors[2])
    #     results.append(result)
    #     print(results)
    # return table


if __name__ == "__main__":

    main("非監督式學習應用", "/mnt/r12921068/DS920/BDS/HW3a/docs/1.pdf")
    # main("keyword", "/mnt/r12921068/DS920/BDS/HW3a/docs/2.pdf")
