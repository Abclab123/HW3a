import camelot
from text2vec import SentenceModel
import streamlit as st
from scipy.spatial.distance import cosine

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        #read titles
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")
        titles = []
        for table in tables:
            text = table.df.to_string().replace("\\n", "").strip().split(' ')
            for word in text:
                if "ai_tables" in word:
                    title = word + " " + text[text.index(word) + 1]
                    title = title.replace("ai_tables_", "")
                    titles.append(title)
        
        #read text
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table, title in zip(tables, titles):
            texts.append({"title": title, "table": table.df, "text": table.df.to_string().replace("\\n", "")})
        return texts
        
class text2vector:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        vector = self.model.encode(text)
        return vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return (1 - cosine(vector_from_table, vector_from_keyword))


def main(keyword, pdf_file):
    #init class
    pdf_parser = pdf2text()
    text_parser = text2vector()
    cos_sim = cosine_sim()
    
    #transform pdf to text
    table_texts = pdf_parser(pdf_file)

    #parse key to vector
    key_vector = text_parser(keyword)

    #calculate cos_sim between text and key
    sims = []
    for table_text in table_texts:
        text = table_text["text"]
        text_vector = text_parser(text)
        sim = cos_sim(key_vector, text_vector)
        sims.append(sim)
        
    # find the max cosine similarity
    max_index = sims.index(max(sims))
    max_table = table_texts[max_index]["table"]
    max_title = table_texts[max_index]["title"]

    return max_table, max_title


if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Table Search Ingine",
        page_icon="random",
        layout="centered",
        initial_sidebar_state="expanded",
        )
    
    st.title("BDS HW3a - 隋中彧")
    st.subheader("Please select a PDF file")
    pdf_file = st.selectbox("PDF file", ["docs/1.pdf", "docs/2.pdf"])
    st.subheader("Please input a keyword")
    keyword = st.text_input("Keyword")
    submit_button = st.button("Search")

    if submit_button or (keyword and pdf_file):
        with st.spinner('The AI is search...'):
            max_table, max_title = main(keyword, pdf_file)
            st.subheader("Result:")
            st.write(max_title)
            st.write(max_table)
