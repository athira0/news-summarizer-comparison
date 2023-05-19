import streamlit as st
from txtai.pipeline import Summary, Textractor
from PyPDF2 import PdfReader
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

@st.cache_resource
def get_bart_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_data
def generate_bart_summary(input):
    summarizer = get_bart_model()
    res = summarizer(input, max_length=130, min_length=30, do_sample=False)
    return res[0]['summary_text']

@st.cache_resource
def get_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
    model = AutoModelForSeq2SeqLM.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
    model.to("cpu")
    return tokenizer,model

# Generate Bert Summary
@st.cache_data
def generate_bert_summary(input):
    tokenizer,model=get_bert_model()
    inputs = tokenizer(input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cpu")
    attention_mask = inputs.attention_mask.to("cpu")
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_str

@st.cache_resource
def get_t5_model():
    tokenizer = AutoTokenizer.from_pretrained("T-Systems-onsite/mt5-small-sum-de-en-v2")
    model = AutoModelForSeq2SeqLM.from_pretrained("T-Systems-onsite/mt5-small-sum-de-en-v2")
    model.to("cpu")
    return tokenizer,model

# Generate t5 Summary
@st.cache_data
def generate_t5_summary(input):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    tokenizer,model=get_t5_model()
    input_ids = tokenizer(input, padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids
    outputs = model.generate(input_ids,max_length=200)
    # all special tokens including will be removed
    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_str

@st.cache_resource
def get_roberta_model():
    tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_cnn_daily_mail")

    model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_cnn_daily_mail")
    model.to("cpu")
    return tokenizer,model

# Generate t5 Summary
@st.cache_data
def generate_roberta_summary(input):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    tokenizer,model=get_roberta_model()
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)[0]
    # all special tokens including will be removed
    output_str = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_str


st.set_page_config(layout="wide")

@st.cache_resource
def text_summary(text, maxlength=None):
    #create summary instance
    summary = Summary()
    text = (text)
    result = summary(text)
    return result

def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text

choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])
get_bert_model()
get_bart_model()
get_roberta_model()
get_t5_model()



if choice == "Summarize Text":
    st.subheader("Summarize the news using a Transformer of your choice")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        checks = st.columns(4)
        with checks[0]:
            model1 = st.checkbox("Bert Summary")
        with checks[1]:
            model2 = st.checkbox('Bart Summary')
        with checks[2]:
            model3 = st.checkbox('Roberta Summary')
        with checks[3]:
            model4 = st.checkbox('t5 Summary')
        if model1:
            sum1=generate_bert_summary(input_text)
            st.success("Bert Summary : "+''.join(str(x) for x in sum1))
        if model2:
            sum2=generate_bart_summary(input_text)
            st.info("Bart Summary : "+''.join(str(x) for x in sum2))
        if model3:
            sum3=generate_roberta_summary(input_text)
            st.warning("Roberta Summary : "+''.join(str(x) for x in sum3))
        if model4:
            sum4=generate_t5_summary(input_text)
            st.error("t5 Summary : "+''.join(str(x) for x in sum4))



elif choice == "Summarize Document":
    st.subheader("Summarize Document using txtai")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])

    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1,1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                st.markdown("**Extracted Text is Below:**")
                st.info(extracted_text)
            with col2:
                st.markdown("**Summary Result**")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)
                
