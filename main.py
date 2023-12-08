import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text

class text2vector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def __call__(self, text):
        return self.vectorizer.fit_transform([text])

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cosine_similarity(vector_from_table, vector_from_keyword)

def search_table_with_keyword(keyword, pdf_file):
    pdf_parser = pdf2text()
    text_converter = text2vector()
    cosine_calculator = cosine_sim()

    table_text = pdf_parser(pdf_file)
    keyword_vector = text_converter(keyword)
    table_vector = text_converter(table_text)

    similarity_score = cosine_calculator(table_vector, keyword_vector)
    
    # Assuming a threshold for similarity (you may adjust this based on your needs)
    similarity_threshold = 0.5

    if similarity_score[0][0] > similarity_threshold:
        tables = camelot.read_pdf(pdf_file, pages="all")
        matching_tables = []
        for i, table in enumerate(tables):
            table_text = table.df.to_string()
            table_vector = text_converter(table_text)
            similarity_score_table = cosine_calculator(table_vector, keyword_vector)
            if similarity_score_table[0][0] > similarity_threshold:
                matching_tables.append(table.df)
        return matching_tables
    else:
        return None

def main(keyword, pdf_file):
    matching_tables = search_table_with_keyword(keyword, pdf_file)

    if matching_tables:
        print(f"Found in '{pdf_file}':")
        for i, table in enumerate(matching_tables):
            print(f"Table {i + 1}:\n{table}\n")
    else:
        print(f"No matching table found in '{pdf_file}' for keyword '{keyword}'.")

if __name__ == "__main__":
    main("desired information", "docs/1.pdf")
    main("desired information", "docs/2.pdf")
