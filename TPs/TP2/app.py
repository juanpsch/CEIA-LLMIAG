import streamlit as st


from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
import re

def decision(prompt):

    indice = None
    persona = None

    jpmatch = re.compile('^.*?(?:Juan|Pablo|Schamun).*$', re.IGNORECASE)
    nicomatch = re.compile('^.*?(?:Nicolas|Cacheda).*$', re.IGNORECASE)

    if jpmatch.match(prompt):
        indice = 'jptest'
        persona = 'Juan Pablo'
        
    elif nicomatch.match(prompt):
        indice = 'nico'
        persona = 'Nicolas'

    return indice, persona




def get_vectorstore(index_name='jptest'):
    embd = OllamaEmbeddings(model="nomic-embed-text")
    
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
    st.set_page_config(page_title="Chat with Juan Pablo's or Nicolás CV", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None



    st.header(f"Chat with Juan Pablo's or Nicolas's CVs PDFs :books:")

    # option = st.selectbox(
    #     "Who do you want to know from?",
    #     ("Juan Pablo", "Nicolas"),
    # )
    
    option = st.text_input(f"Who do you want to know from?")
    indice, persona = decision(option)
    
    if indice:
        user_question = st.text_input(f"Ask a question about {persona}:")   
        if user_question:
            # indice = decision(user_question)            
            handle_userinput(user_question)

        vectorstore = get_vectorstore(index_name=indice)


        # create the conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)               
                
    else:
        st.warning('I can not recognize that person! Try again, please.', icon="⚠️")

if __name__ == '__main__':
     main()
