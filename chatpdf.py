import os
import requests
import pdfplumber
from langchain_groq import ChatGroq
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

from langchain_core.prompts import PromptTemplate

model="your model name"
groq_api_key="your groq api key"

llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)

pdf_path = r"your pdf path"
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

pdf_text = extract_text_from_pdf(pdf_path)

embedder = SpacyEmbeddings(model_name='en_core_web_sm')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=23)
chunks = text_splitter.split_text(pdf_text)

documents = [Document(page_content=chunk) for chunk in chunks]
db = FAISS.from_documents(documents, embedder)



prompt_template = PromptTemplate.from_template(
    """You are an AI Assistant that answers the user question according to the context.
Context: {context}
Question: {question} 
History: {chat_history}
"""
)

retriever = db.as_retriever(search_kwargs={"k": 5})

memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    output_key="answer"
)


while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "end chat":
        print("Goodbye!")
        break

    response = chain.invoke({"question": user_input})
    ai_response = response['chat_history'][-1].content

    while ai_response.strip() == '':
        response = chain.invoke({"question": user_input})
        ai_response = response['chat_history'][-1].content

    print(f"{ai_response}")

print(memory.load_memory_variables({}))
