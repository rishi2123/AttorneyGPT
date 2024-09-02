import streamlit as st
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from typing import List, Dict
import os

# Load environment variables (if any)
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="LawGPT", layout="wide")

# Login to Hugging Face Hub
hf_token = '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'
login(token=hf_token)

# Column setup for the image
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00")

# Custom CSS styling
st.markdown(
    """
    <style>
    .reportview-container { background-color: #1e1e1e; color: #e0e0e0; font-family: 'Arial', sans-serif; }
    .css-1g7bq04 { background-color: #2e2e2e; }
    div.stButton > button:first-child { background-color: #007bff; color: #ffffff; border: none; border-radius: 4px; padding: 10px 20px; }
    div.stButton > button:active { background-color: #0056b3; }
    .css-1v0mbdj { background-color: #2c2c2c; color: #e0e0e0; border-radius: 4px; padding: 10px; }
    .chat-message { border-radius: 10px; padding: 10px; margin-bottom: 10px; max-width: 80%; line-height: 1.5; }
    .chat-message.user { background-color: #007bff; color: #ffffff; align-self: flex-end; }
    .chat-message.assistant { background-color: #333333; color: #e0e0e0; align-self: flex-start; }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    #stDecoration {display: none;}
    button[title="View fullscreen"] {visibility: hidden;}
    .css-18e3th9 { border-radius: 8px; padding: 20px; background-color: #2c2c2c; }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to reset conversation
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Load embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define prompt template
prompt_template = """<s>[INST]This is a chat template and as a legal chatbot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Load model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

class CustomLLM(LLM):
    def __init__(self, generator):
        super().__init__()  # Initialize the base class
        self.generator = generator  # Use a non-private attribute

    def _call(self, prompt: str, stop: List[str] = None, **kwargs) -> str:
        response = self.generator(prompt, max_length=1024, do_sample=True, temperature=0.5)
        return response[0]['generated_text']

    @property
    def _identifying_params(self) -> Dict:
        return {}

    @property
    def _llm_type(self) -> str:
        return "custom"


# Instantiate your custom LLM
custom_llm = CustomLLM(generator)

# Conversational retrieval chain setup with the custom LLM
qa = ConversationalRetrievalChain.from_llm(
    llm=custom_llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display chat history
for message in st.session_state.messages:
    role = message.get("role")
    with st.chat_message(role):
        st.markdown(f'<div class="chat-message {role}">{message.get("content")}</div>', unsafe_allow_html=True)

# Handle user input
input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-message user">{input_prompt}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(f'<div class="chat-message assistant">{full_response} ▌</div>', unsafe_allow_html=True)

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
