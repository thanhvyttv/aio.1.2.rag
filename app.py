import os
import shutil
import sys
import tempfile

# Ensure pysqlite3 is imported and aliased to sqlite3 at the very beginning
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

# Ensure chromadb is imported here, before any Langchain Chroma components
import chromadb  # This is important for PersistentClient to work correctly
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# --- Streamlit Session State Initialization ---
# Initialize session state variables if they don't exist
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None


# --- Model Loading Functions (Cached) ---
# Use st.cache_resource to cache models across reruns for efficiency
@st.cache_resource
def load_embeddings():
    """Loads and caches the HuggingFace Embeddings model."""
    print("Loading embeddings...")
    # Using a smaller, general-purpose embedding model for efficiency
    # If Vietnamese specificity is crucial and resources allow, switch back to 'bkai-foundation-models/vietnamese-bi-encoder'
    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    print("Embeddings loaded!")
    return embeddings


@st.cache_resource
def load_llm():
    """Loads and caches the HuggingFace Language Model (LLM) pipeline."""
    print("Loading LLM...")

    MODEL_NAME = "Vietnamese-Llama2-7B-Chat"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Detect if CUDA (GPU) is available for logging purposes
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}. Using device_map='auto'.")

    # Load the LLM model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,  # Automatically maps model layers to available devices (GPU/CPU)
        # torch_dtype=torch.float16,  # Use float16 for reduced memory footprint
        trust_remote_code=True,  # Necessary for some models if custom code is used
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, trust_remote_code=True
    )

    # Create the text generation pipeline
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        # IMPORTANT: Do NOT specify 'device' here when using 'device_map="auto"' in from_pretrained
        # 'accelerate' already handles device placement, so adding 'device' causes a conflict.
    )

    llm = HuggingFacePipeline(pipeline=model_pipeline)
    print("LLM loaded!")
    return llm


# --- PDF Processing Function ---
def process_pdf(uploaded_file):
    """
    Processes an uploaded PDF file: loads, splits into chunks,
    and stores in a Chroma vector database.
    """
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    print(f"Temporary PDF saved to: {tmp_file_path}")
    st.info("Temporary PDF saved. Loading documents...")

    # Load documents from the PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF.")
    st.info(f"Loaded {len(documents)} pages. Splitting into chunks...")

    # Split documents into semantic chunks using the loaded embeddings
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True,
    )
    docs = semantic_splitter.split_documents(documents)
    print(f"Split PDF into {len(docs)} chunks.")
    st.info(f"Split into {len(docs)} chunks. Initializing ChromaDB...")

    # --- ChromaDB Setup ---
    # Define a temporary directory for ChromaDB persistence
    persist_directory = os.path.join(
        tempfile.gettempdir(), "chroma_db_langchain_rag_app"
    )
    print(f"ChromaDB persist directory: {persist_directory}")
    st.info(f"ChromaDB will persist at: {persist_directory}")

    # Remove the directory if it already exists to ensure a fresh start
    # This is crucial on Streamlit Cloud as temp directories persist across app restarts (but not new deploys)
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print(f"Successfully removed old ChromaDB directory: {persist_directory}")
        except OSError as e:
            # Handle cases where directory might be in use or have permission issues
            print(
                f"Error removing directory {persist_directory}: {e}. Proceeding anyway."
            )

    # Create the directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize a PersistentClient and pass it to Chroma.from_documents
    # This ensures Chroma uses the pysqlite3 backend correctly
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    print("ChromaDB client initialized.")

    # Create the Chroma vector database from the document chunks
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=st.session_state.embeddings,
        client=chroma_client,  # Pass the initialized client here
        collection_name="my_pdf_rag_collection",  # Give your collection a distinct name
    )
    print("Vector database created.")

    # Create a retriever from the vector database
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    # Pull the RAG prompt from Langchain Hub
    prompt = hub.pull("rlm/rag-prompt")
    print("RAG prompt pulled from Langchain Hub.")

    # Define a function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Construct the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )
    print("RAG chain constructed.")

    # Clean up the temporary PDF file
    os.unlink(tmp_file_path)
    print(f"Temporary PDF deleted: {tmp_file_path}")

    return rag_chain, len(docs)


# --- Streamlit UI Definition ---
def main():
    """Main function to define the Streamlit application UI."""
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

    # Conditional loading of models: only load once
    if not st.session_state.models_loaded:
        st.info("Loading models.... This may take a moment.")
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Models are ready!")
        # Rerun the app after models are loaded to clean up the 'st.info' message
        st.rerun()
        # Note: st.rerun() will restart the script, so the code below this block
        # will only run on the subsequent execution where models_loaded is True.

    # Only show file uploader and Q&A if models are loaded
    if st.session_state.models_loaded:
        # File Uploader and Processing Button
        uploaded_file = st.file_uploader("Upload file PDF", type="pdf")
        if uploaded_file and st.button("Processing PDF"):
            with st.spinner("Processing PDF and creating knowledge base..."):
                st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                st.success(f"Finished! Processed {num_chunks} chunks from PDF.")

        # Question & Answer Section
        if st.session_state.rag_chain:
            question = st.text_input("Make a question about the PDF content:")
            if question:
                with st.spinner("Generating answer..."):
                    try:
                        output = st.session_state.rag_chain.invoke(question)
                        # Attempt to parse the answer if "Answer:" is present
                        answer = (
                            output.split("Answer:")[1].strip()
                            if "Answer:" in output
                            else output.strip()
                        )
                        st.write("**Answer:**")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred during answering: {e}")
                        st.warning(
                            "Please try asking another question or re-uploading the PDF."
                        )
                        print(f"Error during RAG chain invocation: {e}")


# --- Entry Point ---
if __name__ == "__main__":
    main()
