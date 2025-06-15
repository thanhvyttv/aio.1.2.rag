# import torch
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_huggingface.llms import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- CẤU HÌNH CÁC BIẾN CƠ BẢN ---
# Đường dẫn file PDF của bạn
FILE_PATH = "./data/YOLOv10_Tutorials.pdf"

# Tên mô hình LLM. Lưu ý: Vicuna-7b-v1.5 là mô hình lớn (khoảng 14GB ở FP16),
# đảm bảo bạn có đủ RAM trên MacBook Intel để chạy nó ở FP32/FP16/BF16.
# Nếu gặp lỗi out of memory, bạn cần chọn mô hình nhỏ hơn.
# MODEL_NAME = "lmsys/vicuna-7b-v1.5"

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


# --- KHỞI TẠO MÔ HÌNH NGÔN NGỮ LỚN (LLM) ---
print("--- Đang khởi tạo mô hình ngôn ngữ lớn (LLM) ---")

# --- KHỞI TẠO MÔ HÌNH NGÔN NGỮ LỚN (LLM) VỚI LLAMA-CPP-PYTHON ---
print("--- Đang khởi tạo mô hình ngôn ngữ lớn (LLM) với LlamaCpp ---")

# Đường dẫn đến file GGUF bạn đã tải xuống
# Hãy thay thế bằng đường dẫn thực tế của bạn
GGUF_MODEL_PATH = "./data/models/vicuna-7b-v1.5.Q4_K_M.gguf"

# Cấu hình callback manager để in output ra console ngay lập tức
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=GGUF_MODEL_PATH,
    temperature=0.75,
    max_tokens=512,
    n_ctx=2048,
    n_gpu_layers=0,  # QUAN TRỌNG: Đặt 0 để đảm bảo chỉ chạy trên CPU
    n_batch=512,  # Kích thước batch để xử lý song song các token
    callback_manager=callback_manager,
    verbose=False,  # Đặt True để xem thông tin tải mô hình chi tiết hơn
)

# # Tích hợp tokenizer và model thành một pipeline
# # device_map="cpu" là cần thiết để đảm bảo không cố gắng sử dụng GPU
# model_pipline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     pad_token_id=tokenizer.eos_token_id,
#     device_map="cpu",  # Quan trọng: chỉ định rõ ràng chạy trên CPU
# )

# # Khởi tạo HuggingFacePipeline cho LangChain
# llm = HuggingFacePipeline(
#     pipeline=model_pipline,
# )

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
