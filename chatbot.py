import os
import base64
import streamlit as st
from PIL import Image
from streamlit_chat import message
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from groq import Groq

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=api_key)

# Vector Store Setup
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
else:
    st.error("Vector store not found. Please run the vector store creation code first.")

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{get_base64_encoded_image('C:/Users/Ritesh/Desktop/lumi/gradient-bg.jpg')}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: #1E1E1E;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



def groq_llm(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a compassionate and caring mental health therapist. Your goal is to provide thoughtful, supportive, and helpful guidance to users who are seeking advice and assistance regarding their mental health. Respond to their queries with empathy, love, and a deep understanding of mental health concerns."
            },
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=1024
    )
    return chat_completion.choices[0].message.content


class GroqLLM(LLM):
    def _call(self, prompt, stop=None):
        return groq_llm(prompt)

    @property
    def _identifying_params(self):
        return {"name": "GroqLLM"}

    @property
    def _llm_type(self):
        return "custom"

# Initialize Components
groq_llm_wrapper = GroqLLM()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(
    llm=groq_llm_wrapper,
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

def set_styles():
    st.markdown("""
        <style>
        /* Root Variables */
        :root {
            --chat-bg-color: #1e1e2f; /* Background color for chat box */
            --text-color: #fafafa; /* Text color */
        }

        /* Title Styling */
        .title {
            font-family: 'Comfortaa', cursive; /* Modern, friendly font */
            font-size: 3.5rem;
            color: var(--text-color);
            text-align: center;
            margin-bottom: 40px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        /* Chat History Container */
        .stChatMessageBox {
            background-color: var(--chat-bg-color); /* Background color for chat history box */
            border-radius: 15px; /* Rounded corners */
            padding: 20px; /* Inner padding */
            margin: 10px 0; /* Outer spacing */
            overflow-y: auto; /* Scroll if content overflows */
            height: 500px; /* Fixed height for the chat box */
            display: flex;
            flex-direction: column; /* Stack messages vertically */
            gap: 10px; /* Space between messages */
        }

        /* Individual Chat Message */
        .stChatMessage {
            background-color: rgba(0, 0, 0, 0.3); /* Semi-transparent message background */
            color: var(--text-color);
            border-radius: 10px;
            padding: 10px;
            margin: 0; /* Remove gaps around messages */
        }

        /* User Input Box */
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: white;
            padding: 10px;
        }
        
        /* Send Button Styling */
        .stButton button {
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: rgba(255, 255, 255, 0.3);
        }
        
        </style>
    """, unsafe_allow_html=True)




def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi, I'm Lumi! I'm here to support you. Feel free to ask me anything related to mental health—I'm here to listen and help."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query):
    formatted_query = prompt.format(question=query)
    result = chain({"question": formatted_query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def display_chat_history():
    # Container for chat history
    chat_history_container = st.container()
    with chat_history_container:
        for i in range(len(st.session_state["generated"])):
            # User's message
            st.markdown(
                f"""
                <div style="background-color:#B2B5E0; color:#1D1842; padding:10px; border-radius:10px; margin:10px 0; max-width:80%; align-self:flex-end;">
                    <strong>You:</strong> {st.session_state["past"][i]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Lumi's response
            st.markdown(
                f"""
                <div style="background-color:#C5ADC5; color:#1D1842; padding:10px; border-radius:10px; margin:10px 0; max-width:80%; align-self:flex-start;">
                    <strong>Lumi:</strong> {st.session_state["generated"][i]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Input form for user query (below the chat history)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Message Lumi...",
            key="chat_input",
            placeholder="Type your message here...",
        )
        submit_button = st.form_submit_button("Send")

    # Process user input and generate a response
    if submit_button and user_input:
        output = conversation_chat(user_input)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)


def main():
    st.set_page_config(
        page_title="Lumi - AI Therapist",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    set_background()
    set_styles()
    
    st.markdown("<h1 class='title'>Lumi - Your AI Therapist</h1>", unsafe_allow_html=True)
    
    initialize_session_state()
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        display_chat_history()

if __name__ == "__main__":
    main()