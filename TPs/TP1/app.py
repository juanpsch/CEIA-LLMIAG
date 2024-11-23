import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama.chat_models import ChatOllama
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq




def get_vectorstore():
    embd = OllamaEmbeddings(model="nomic-embed-text")
    
    index_name = 'jptest'
    namespace = "espacio"

    vectorstore = PineconeVectorStore(index_name=index_name,embedding=embd,
                                          namespace=namespace,)

    return vectorstore


def get_conversation_chain(vectorstore):

    llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():

    load_dotenv()
    st.set_page_config(page_title="Chat with Juan Pablo's CV", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about Juan Pablo:")
    if user_question:
        handle_userinput(user_question)


    vectorstore = get_vectorstore()

    # create the conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore)               
                
    

if __name__ == '__main__':
     main()
