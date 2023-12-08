import tabula
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class pdf2text():
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        # Extract tables from the PDF
        tables = tabula.read_pdf(pdf_file, pages='all', multiple_tables=True)
        # Convert tables to text

        text_tables = [table.to_string() for table in tables]
        return text_tables
from sklearn.feature_extraction.text import TfidfVectorizer

class text2vector():
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def __call__(self, texts):
        return self.vectorizer.fit_transform(texts)
from sklearn.metrics.pairwise import cosine_similarity

class cosine_sim():
    def __init__(self):
        pass

    def __call__(self, vector1, vector2):
        return cosine_similarity(vector1, vector2)
def main(keyword, pdf_file):
    # Initialize classes
    extractor = pdf2text()
    vectorizer = text2vector()
    similarity = cosine_sim()

    # Extract text from tables in the PDF
    text_tables = extractor(pdf_file)

    # Convert text and keyword to vectors
    text_vectors = vectorizer(text_tables + [keyword])
    keyword_vector = text_vectors[-1]  # Last vector is the keyword

    # Compute similarity scores
    scores = similarity(text_vectors[:-1], keyword_vector)

    # Find the table with the highest score
    best_match_index = scores.argmax()
    return text_tables[best_match_index]
table = main("your_keyword", "path_to_your_pdf.pdf")
print(table)