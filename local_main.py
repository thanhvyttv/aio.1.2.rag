import torch
from langchain import hub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import (  # BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Đọc file PDF
Loader = PyPDFLoader
FILE_PATH = "./data/YOLOv10_Tutorials.pdf"
loader = Loader(FILE_PATH)
documents = loader.load()

# Khởi tạo mô hình embedding
embedding = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# Khởi tại bộ tách văn bản
semantic_splitter = SemanticChunker(
    embeddings=embedding,
    buffer_size=1,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
    min_chunk_size=500,
    add_start_index=True,
)

docs = semantic_splitter.split_documents(documents)
print("Number of semantic chunks: ", len(docs))

# Khởi tạo vector database
vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
retriever = vector_db.as_retriever()
result = retriever.invoke("What is YOLO?")

print("Number of relevent documents: ", len(result))

# Khởi tạo mô hình ngôn ngữ lớn
# Khai  báo một số cài đặt cần thiết cho mô hình
# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )


# Khởi tạo mô hình và tokenizer:
# MODEL_NAME = "lmsys/vicuna-7b-v1.5"
MODEL_NAME = "google/gemma-2b-it"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

model.to("cpu")

# Tích hợp tokenizer và model thành một pipeline để tiện sử dụng
model_pipline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto",
)

llm = HuggingFacePipeline(
    pipeline=model_pipline,
)

# Chạy chương trình
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

print(answer)
