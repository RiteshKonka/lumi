from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from streamlit_chat import message
from langchain_community.llms import CTransformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate  # Import for prompt templates
import os

if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True  # Enable this flag
    )
else:
    st.error("Vector store not found. Please run the vector store creation code first.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':512,'temperature':0.3})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

prompt_template = """
You are a compassionate and caring mental health therapist. Your goal is to provide thoughtful, supportive, and helpful guidance to users who are seeking advice and assistance regarding their mental health. Respond to their queries with empathy, love, and a deep understanding of mental health concerns:

Question: {question}
"""


prompt = PromptTemplate(
    input_variables=["question"], 
    template=prompt_template
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&family=Open+Sans:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# Injecting custom CSS to style the title, input field, and chat bubbles
st.markdown("""
    <style>
    /* Title Styling */
    h1 {
        font-family: 'Coming Soon', cursive;
        font-size: 3rem;
        color: #33ccff;  /* Soft blue */
        font-weight: bold;
        margin-bottom: 70px;
        margin-top: 10px;  /* Reduces space above the title */
        padding-top: 0;  /* Removes any padding that might be pushing it down */
    }

    /* Background Styling */
    body {
        font-family: 'Open Sans', sans-serif;
        background: linear-gradient(180deg, #121212 0%, #1E1E1E 100%); /* Dark gradient background */
        color: #FFFFFF;  /* White for text */
    }

    /* Chat Bubble Styling */
    .stMessageUser {
        background-color: #357ABD;  /* Darker blue for user message bubbles */
        color: #FFFFFF;  /* White text for better contrast */
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        font-family: 'Open Sans', sans-serif;
        font-size: 1rem;
    }
    
    .stMessageAI {
        background-color: #2C2C2C;  /* Light gray for AI message bubbles */
        color: #FFFFFF;  /* White text */
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        font-family: 'Open Sans', sans-serif;
        font-size: 1rem;
    }

    /* Input Placeholder and Field Styling */
    input {
        font-family: 'Open Sans', sans-serif;
        font-size: 1rem;
        color: #B0BEC5;  /* Light gray for placeholder text */
        border: 2px solid #33ccff;  /* Soft blue border */
        padding: 10px;
        border-radius: 8px;
        background-color: #1E1E1E;  /* Darker background for input */
    }

    /* Chatbox and Send Button Styling */
    .css-1q8dd3e {
        background-color: #33ccff;  /* Light blue background for send button */
        color: white;
        font-family: 'Lato', sans-serif;
        font-size: 1rem;
        border-radius: 8px;
        padding: 10px 20px;
        cursor: pointer;
    }
    
    .css-1q8dd3e:hover {
        background-color: #357ABD;  /* Darker blue on hover */
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1E1E1E;
    }

    ::-webkit-scrollbar-thumb {
        background: #33ccff; /* Soft blue for scrollbar */
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #357ABD;  /* Darker blue on hover */
    }
    </style>
    """, unsafe_allow_html=True)



# Streamlit app content here
st.title("Lumi - Your AI Therapist")

def conversation_chat(query):
    # Format the query using the prompt template
    formatted_query = prompt.format(question=query)
    
    # Pass the formatted query into the chain
    result = chain({"question": formatted_query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi, I'm Lumi! I'm here to support you. Feel free to ask me anything related to mental health—I'm here to listen and help."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Mental Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()
