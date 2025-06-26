import os
import shutil
import sys
import tempfile

# Đảm bảo các dòng này là những dòng đầu tiên của file, trước các imports khác.
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Cần import chromadb sớm để các cài đặt có hiệu lực
import chromadb  # <--- Đảm bảo import chromadb ở đây
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

# from chromadb.config import Settings # Không cần nếu chỉ dùng PersistentClient


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
    print("Loading embeddings...")
    # Sử dụng một embedding model nhỏ hơn, phổ biến hơn và đa ngôn ngữ tốt hơn
    # 'bkai-foundation-models/vietnamese-bi-encoder' có thể cần tài nguyên lớn
    # và có thể không được tối ưu cho tốc độ/bộ nhớ trên cloud miễn phí
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    print("Embeddings loaded!")
    return embeddings


@st.cache_resource
def load_llm():
    print("Loading LLM...")
    # CHỌN MỘT MÔ HÌNH NHỎ HƠN ĐỂ TRÁNH LỖI OOM
    # TinyLlama là 1.1B params, vẫn có thể nặng trên Streamlit Cloud
    # DistilGPT2 là 124M params, nhẹ hơn rất nhiều, dùng để test chức năng
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Hoặc "distilgpt2" nếu TinyLlama vẫn chết queo

    # KHI SỬ DỤNG CUDA (GPU)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # quantization_config=bnb_config, # Đã comment, rất tốt
        device_map="auto",  # Tự động map sang GPU/CPU
        torch_dtype=torch.float16,  # SỬ DỤNG float16 ĐỂ TIẾT KIỆM BỘ NHỚ
        # offload_folder="./offload", # LOẠI BỎ: Gây chậm và có thể lỗi RAM
        trust_remote_code=True,  # Giữ lại nếu model yêu cầu
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
        device=device,  # CHỈ ĐỊNH DEVICE RÕ RÀNG
    )

    llm = HuggingFacePipeline(pipeline=model_pipeline)
    print("LLM loaded!")
    return llm


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

    # CẤU HÌNH VÀ KHỞI TẠO CHROMADB LẠI CHO ĐÚNG CÁCH
    persist_directory = os.path.join(tempfile.gettempdir(), "chroma_db_langchain")
    print(f"ChromaDB persist directory: {persist_directory}")

    # Xóa thư mục cũ để đảm bảo một khởi đầu sạch sẽ
    # Quan trọng cho việc debug lỗi sqlite3/no such table
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print(f"Successfully removed old ChromaDB directory: {persist_directory}")
        except OSError as e:
            print(
                f"Error removing directory {persist_directory}: {e}. Trying to proceed."
            )
            # Nếu xóa không được, có thể thử đổi tên thư mục để Chroma tạo cái mới
            # Hoặc chấp nhận rằng việc persist có thể không hoạt động hoàn hảo

    os.makedirs(persist_directory, exist_ok=True)

    # Khởi tạo một Chroma client rõ ràng và truyền vào
    # Đây là điểm mấu chốt để giải quyết lỗi sqlite3/no such table
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=st.session_state.embeddings,
        client=chroma_client,  # <--- TRUYỀN CLIENT ĐÃ KHỞI TẠO Ở ĐÂY
        collection_name="my_pdf_rag_collection",  # Nên đặt tên rõ ràng cho collection
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
