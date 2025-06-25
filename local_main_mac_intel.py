import torch
from langchain import hub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# --- CẤU HÌNH CÁC BIẾN CƠ BẢN ---
# Đường dẫn file PDF của bạn
FILE_PATH = "./data/YOLOv10_Tutorials.pdf"

# --- ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ---
print("--- Đang đọc file PDF và chia nhỏ tài liệu ---")
loader = PyPDFLoader(FILE_PATH)
documents = loader.load()

# Khởi tạo mô hình embedding (để tạo vector nhúng cho văn bản)
# Sử dụng mô hình đã được xác định hoạt động tốt trên CPU
embedding = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# Khởi tạo bộ tách văn bản ngữ nghĩa
semantic_splitter = SemanticChunker(
    embeddings=embedding,
    buffer_size=1,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
    min_chunk_size=500,
    add_start_index=True,
)

docs = semantic_splitter.split_documents(documents)
print(f"Số lượng chunk ngữ nghĩa: {len(docs)}")

# Khởi tạo vector database (ChromaDB)
# Đây là nơi lưu trữ các vector nhúng của tài liệu để tìm kiếm nhanh chóng
print("--- Đang khởi tạo Vector Database ---")
vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vector_db.as_retriever()

# Thử truy vấn để kiểm tra retriever
result = retriever.invoke("YOLO là gì?")
print(f"Số lượng tài liệu liên quan được tìm thấy: {len(result)}")

MODEL_PATH = "./models/vicuna-7b-v1.5.Q4_K_M.gguf"

# Cấu hình callback manager để in output ra console ngay lập tức
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.75,
    max_tokens=2000,
    n_ctx=4096,
    n_gpu_layers=0,
    verbose=True,
)

print("Mô hình LlamaCpp đã được khởi tạo thành công.")
# --- CHẠY CHƯƠNG TRÌNH RAG ---
print("--- Đang chạy truy vấn RAG ---")
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

USER_QUESTION = "YOLOv10 là gì?"
output = rag_chain.invoke(USER_QUESTION)
answer = output.split("Answer:")[1].strip()

print("\n--- Câu trả lời từ RAG ---")
print(answer)
