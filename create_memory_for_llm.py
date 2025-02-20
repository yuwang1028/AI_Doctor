import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ ERROR: OpenAI API Key is missing! Set 'OPENAI_API_KEY' in your environment.")

# Define data and FAISS storage paths
DATA_PATH = "data/"
FAISS_DB_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_dir: str):
    """Loads all PDF files in the specified directory using PyPDFLoader."""
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Data directory '{data_dir}' not found.")
        return []

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ WARNING: No PDF files found in the directory.")
        return []

    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents from {data_dir}")
    return documents

def create_chunks(extracted_data):
    """Splits documents into smaller text chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"✅ Created {len(text_chunks)} text chunks.")
    return text_chunks

def check_existing_faiss():
    """Checks if an existing FAISS database is available."""
    if os.path.exists(FAISS_DB_PATH):
        print("🔍 FAISS vector store already exists. Overwriting...")
        return True
    return False

def main():
    # 1. Load PDF documents
    documents = load_pdf_files(DATA_PATH)
    if not documents:
        print("❌ ERROR: No documents to process. Exiting...")
        return

    # 2. Create text chunks
    text_chunks = create_chunks(documents)
    
    # 3. Load embedding model
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    test_vector = embedding_model.embed_query("Test query")  # Ensure embedding dimension matches FAISS
    print(f"✅ OpenAI Embedding Model Loaded (Vector Dimension: {len(test_vector)})")

    # 4. Create FAISS vector store (overwrite if exists)
    check_existing_faiss()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(FAISS_DB_PATH)
    print("✅ FAISS vector store created and saved successfully.")

if __name__ == "__main__":
    main()

