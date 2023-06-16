import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
import os, fitz
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI, VectorDBQA
# from dotenv import load_dotenv
import openai
from typing import Any, List, Optional
from langchain.docstore.document import Document
# load_dotenv()

st.title("Q&A Chat Bot using GPT3")

openai.api_key = os.getenv('OPEN_AI_API_KEY')

def load(self, **kwargs: Optional[Any]) -> List[Document]:
        """Load file."""

        doc = fitz.open("pdf",self.getvalue())  # open document
        # file_path = self.file_path if self.web_path is None else self.web_path

        return [
            Document(
                page_content=page.get_text(**kwargs).encode("utf-8"),
                metadata=dict(
                    {
                        "source": "Uploaded",
                        "file_path": "N/A",
                        "page_number": page.number + 1,
                        "total_pages": len(doc),
                    },
                    **{
                        k: doc.metadata[k]
                        for k in doc.metadata
                        if type(doc.metadata[k]) in [str, int]
                    }
                ),
            )
            for page in doc
        ]




def extract_pdf(pdf_filepath, question):
    documents = load(pdf_filepath) 
    # st.write(type(documents[0]))
    text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000) 
    texts = text_splitter.split_documents(documents) 
    # print(texts) 
    llm = OpenAI(model_name="text-davinci-003", temperature=0.7,openai_api_key=openai.api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key) 
    docsearch = Chroma.from_documents(texts, embeddings)
    # qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch) 
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}), ) 
    answer = qa.run(question) 
    return answer

# def extract_pdf(pdf_file, question):
#     # loader = PyMuPDFLoader(pdf_filepath) 
#     document = fitz.open("pdf",pdf_file.getvalue())
#     template = {'page_content':"some text", 'lookup_index':0}
#     i = 0
#     documents=[]
#     for page in document:
#         temp = template.copy()
#         temp['page_content'] = page.get_text()
#         temp['lookup_index'] = i
#         i+=1
#         documents.append(temp)
#     text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000) 
#     texts = text_splitter.split_documents(documents) 
#     # print(texts) 
#     llm = OpenAI(model_name="text-davinci-003", temperature=0.7,openai_api_key=openai.api_key)
#     embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key) 
#     docsearch = Chroma.from_documents(texts, embeddings)
#     # qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch) 
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}), ) 
#     answer = qa.run(question) 
#     return answer

pdf_file = st.file_uploader("Give the file here")
# st.write(pdf_file)
if pdf_file is not None:
    with st.form('Q&A Form'):
        question = st.text_input("Ask your question",placeholder="You question ...")
        # st.text_area("**AI's** Answer is ",value=extract_pdf(pdf_file, question))
        ask = st.form_submit_button("Ask")
        if ask:
            st.text_area("**AI's** Answer is ",value=extract_pdf(pdf_file, question))
            # st.text_area("**AI's** Answer is ",value=extract_pdf("/home/laptop-obs-135/Downloads/Hand_Book-Assessment_and_Management_of_Mental_Health_Problems_in_General_Practice.pdf", question))
