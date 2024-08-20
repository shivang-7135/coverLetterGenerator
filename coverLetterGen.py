import proto
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
tes = os.getenv("GOOGLE_API_KEY")
print(tes)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Initialize embeddings (if needed)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def main():

    st.title("Cover Letter Generator")
    job_description_input = st.text_input("Enter job description")
    Resume_input = st.text_input("Enter your Resume")
    if st.button("Generate Cover Letter"):
        prompt = f"Provide a Cover letter using this job description and Resume which is concise but beutifully created in 150 words. Make sure it is more humanly and less bot written. {job_description_input} {Resume_input}"
        response = model.predict(prompt)
        print(response)
        st.write(response)



if __name__ == "__main__":
    main()