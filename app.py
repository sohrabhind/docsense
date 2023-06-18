import streamlit as st
from PIL import Image
from transformers import pipeline
from PyPDF2 import PdfReader

st.set_page_config(
    page_title="DocSense",
    layout="centered",
    page_icon="static/logo.png",
    initial_sidebar_state="auto",
)

main_image = Image.open('static/banner.png')

@st.cache_data(show_spinner=True)
def instantiate_pipe():
    model_name = "deepset/roberta-base-squad2"
    print("Loading model: ", model_name)
    pipe = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return pipe


st.image(main_image, use_column_width='auto')
st.markdown("<strong>AI-driven answers from your documents.</strong>", unsafe_allow_html=True)
pipe = instantiate_pipe()
uploaded_file = st.file_uploader("Upload document file", type=['pdf', 'txt'])

if uploaded_file is not None:
    context = ""
    if uploaded_file.name.split(".")[-1].lower() == "pdf":
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            context += page.extract_text().strip() + "\n"
    elif uploaded_file.name.split(".")[-1].lower() == "txt":
        for line in uploaded_file:
            context += line.decode("utf-8").strip() + "\n"
    context = context.replace("\n", " ")
    context = context.replace("\r", " ")

    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    text_input = st.text_input(
        "Enter your question 👇",
    ) 
    if text_input:
        with st.spinner(f"Working..."):
            qa_input = {
                'question': text_input,
                'context': context
            }
            pipe_response = pipe(qa_input)
            st.success("Response: "+pipe_response['answer'])
else:
    st.markdown("<center><strong>OR</strong></center>", unsafe_allow_html=True)
    context = st.text_area(
        "Enter your content 👇",
    )
    if context:
        context = context.replace("\n", " ")
        context = context.replace("\r", " ")
        text_input = st.text_input(
        "Enter your question 👇",
        )
        if text_input:
            with st.spinner(f"Working..."):
                qa_input = {
                    'question': text_input,
                    'context': context
                }
                pipe_response = pipe(qa_input)
                st.success("Response: "+pipe_response['answer'])

st.markdown("<br><hr><center>Made by <a href=''><strong>Sohrab</strong></a></center><hr>", unsafe_allow_html=True)
st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)