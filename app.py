import os
import tempfile

import streamlit as st
import torch
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Session state initialization
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None


# Load embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )


# Load LLM without bitsandbytes
@st.cache_resource
def load_llm():
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        offload_folder="./offload",  # Bắt buộc nếu thiếu VRAM
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, trust_remote_code=True
    )

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        # Không cần device nếu device_map="auto"
    )

    return HuggingFacePipeline(pipeline=model_pipeline)


# Process uploaded PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True,
    )

    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(
        documents=docs, embedding=st.session_state.embeddings
    )
    retriever = vector_db.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    os.unlink(tmp_file_path)
    return rag_chain, len(docs)


# Streamlit UI
def main():
    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
    st.title("PDF RAG Assistant")

    st.markdown(
        """
    **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**

    **Cách sử dụng đơn giản:**
    1. **Upload PDF** → Chọn file PDF từ máy tính và nhấn "Xử lý PDF"  
    2. **Đặt câu hỏi** → Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức

    ---
    """
    )

    if not st.session_state.models_loaded:
        st.info("Đang tải models...")
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Models đã sẵn sàng!")
        st.rerun()

    uploaded_file = st.file_uploader("Upload file PDF", type="pdf")
    if uploaded_file and st.button("Xử lý PDF"):
        with st.spinner("Đang xử lý..."):
            st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
            st.success(f"Hoàn thành! {num_chunks} chunks")

    if st.session_state.rag_chain:
        question = st.text_input("Đặt câu hỏi:")
        if question:
            with st.spinner("Đang trả lời..."):
                output = st.session_state.rag_chain.invoke(question)
                answer = (
                    output.split("Answer:")[1].strip()
                    if "Answer:" in output
                    else output.strip()
                )
                st.write("**Trả lời:**")
                st.write(answer)


if __name__ == "__main__":
    main()
