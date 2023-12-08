import camelot
from text2vec import SentenceModel
from sentence_transformers.util import cos_sim

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df)
        
        for i in range(len(texts)):
            texts[i] = texts[i].replace('\n', '', regex=True)
        
        return texts


class text2vector:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cos_sim(vector_from_table, vector_from_keyword)

def find_table(keyword, pdf_file):
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file)
    transformer = text2vector()
    vectors , vv = transformer(tables, keyword)
    keyword_vector = vv.transform([keyword])
    table_vectors = vectors
    cos = cosine_sim()
    count = cos(table_vectors, keyword_vector)

    if len(count) > 0:
        index = np.argmax(count)
        if count[index] > 0:
            print(tables[index])
        else:
            print("No table found.")
    else:
        print("No table found.")

def main(keyword, pdf_file):
    parser = argparse.ArgumentParser(description="Find table with keyword.")
    parser.add_argument("keyword", type=str, help="Keyword")
    parser.add_argument("pdf_file", type=str, help="File name")
    args = parser.parse_args()
    find_most_relevant_table(args.keyword, args.pdf_file)

if __name__ == "__main__":
    main()
