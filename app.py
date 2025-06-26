import os
import shutil
import sys
import tempfile

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]


import chromadb
import streamlit as st
import torch
from chromadb.config import Settings
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import (  # BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Session state initialization
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None


# Function
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )


@st.cache_resource
def load_llm():
    # MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_quant_type="nf4",
    # )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )

    return HuggingFacePipeline(pipeline=model_pipeline)


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

    # Định nghĩa persist_directory cho Chroma (nếu bạn muốn lưu trữ cục bộ)
    persist_directory = os.path.join(tempfile.gettempdir(), "chroma_db_langchain")

    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print(f"Removed old ChromaDB directory: {persist_directory}")
        except OSError as e:
            # Điều này xảy ra nếu thư mục đang bị khóa hoặc có vấn đề về quyền
            print(f"Error removing directory {persist_directory}: {e}")
            # Thử một tên thư mục khác nếu xóa thất bại, để đảm bảo khởi đầu mới
            persist_directory = os.path.join(
                tempfile.gettempdir(), f"chroma_db_langchain_{os.getpid()}"
            )
            os.makedirs(persist_directory, exist_ok=True)
    else:
        os.makedirs(persist_directory, exist_ok=True)

    # CẤU HÌNH CLIENT VỚI SETTINGS ĐỂ BUỘC SỬ DỤNG DUCKDB
    settings = Settings(
        persist_directory=persist_directory,
        is_persistent=True,  # Đảm bảo chế độ persistent
    )

    # Khởi tạo client
    client = chromadb.PersistentClient(path=persist_directory)

    # Sử dụng client này khi khởi tạo Chroma
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=st.session_state.embeddings,
        client=client,
        collection_name="my_pdf_collection",
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


# UI
def main():
    st.set_page_config(page_title="PDF RAG Assisstant", layout="wide")
    st.title("PDF RAG Assisstant")

    st.markdown(
        """
        **Ứng dụng AI giúp hỏi đáp trực tiếp với nội dung PDF bằng tiếng Việt**

        **Cách sử dụng:**
        1. **Upload file PDF** → chọn file PDF
        2. **Đặt câu hỏi**

        ----
        """
    )

    # Load models:
    if not st.session_state.models_loaded:
        st.info("Loading models....")
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Models is ready!")
        st.rerun()

    # Upload PDF
    uploaded_file = st.file_uploader("Upload file PDF", type="pdf")
    if uploaded_file and st.button("Processing PDF"):
        with st.spinner("Processing"):
            st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
            st.success(f"Finish! {num_chunks} chunks")

    # Q&A
    if st.session_state.rag_chain:
        question = st.text_input("Make a question:")
        if question:
            with st.spinner("Answering...."):
                output = st.session_state.rag_chain.invoke(question)
                answer = (
                    output.split("Answer:")[1].strip()
                    if "Answer:" in output
                    else output.strip()
                )
                st.write("**Answer**")
                st.write(answer)


if __name__ == "__main__":
    main()
