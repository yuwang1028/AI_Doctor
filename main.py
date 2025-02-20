import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from fastapi.middleware.cors import CORSMiddleware

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure OpenAI API Key is loaded
if not OPENAI_API_KEY:
    raise ValueError("❌ ERROR: OpenAI API Key is missing! Set 'OPENAI_API_KEY' in your environment.")
else:
    print(f"✅ OpenAI API Key Loaded: {OPENAI_API_KEY[:6]}... (Masked)")

app = FastAPI(title="AI Chatbot API", description="A chatbot API using OpenAI and FAISS", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all request methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all request headers
)
# Path to FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    """Loads the FAISS vector store with OpenAI embeddings and checks dimensions."""
    try:
        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        print(f"✅ Using OpenAI Embeddings Model")

        # Load FAISS index
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

        # Check FAISS Index Dimension
        index = vectorstore.index
        print(f"✅ FAISS Index Dimension: {index.d}")

        # Verify embedding model produces the same dimension
        test_vector = embedding_model.embed_query("Test query")
        print(f"✅ Query Embedding Dimension: {len(test_vector)}")

        if index.d != len(test_vector):
            print("❌ FAISS dimension mismatch! Rebuilding FAISS...")
            rebuild_faiss(embedding_model)

        return vectorstore
    except Exception as e:
        print(f"❌ ERROR: Failed to load FAISS vector store: {str(e)}")
        return None

def rebuild_faiss(embedding_model):
    """Rebuilds FAISS index to ensure correct embedding dimensions."""
    print("🔄 Rebuilding FAISS Index...")
    texts = ["Medical knowledge about fever", "Treatment for cold symptoms", "How to prevent headaches"]
    db = FAISS.from_texts(texts, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("✅ FAISS Index Rebuilt Successfully!")

def load_llm():
    """Loads OpenAI GPT model."""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=512,
        openai_api_key=OPENAI_API_KEY
    )

# Define prompt template
system_prompt = (
    "you are the expert of medical"
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use five sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create retrieval chain
def create_retrieval_chain():
    """Creates the retrieval-based LLM pipeline."""
    vectorstore = get_vectorstore()
    if not vectorstore:
        raise ValueError("❌ ERROR: FAISS vector store could not be loaded!")

    retriever = VectorStoreRetriever(vectorstore=vectorstore)

    retrieval_chain = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ) | prompt | load_llm()

    return retrieval_chain

# Request model
class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(query_request: QueryRequest):
    """Handles user queries and returns AI-generated responses."""
    try:
        query_text = query_request.query.strip()  # Ensure it's a clean string

        print(f"📩 Received Query: {query_text}")

        retrieval_chain = create_retrieval_chain()
        result = retrieval_chain.invoke(query_text)

        # Extract response text correctly
        if isinstance(result, dict) and "content" in result:
            response_text = result["content"]
        else:
            response_text = str(result)

        print(f"📤 Response: {response_text}")
        return {"response": response_text}

    except Exception as e:
        error_message = f"❌ Internal Server Error: {str(e)}\n{traceback.format_exc()}"
        print(error_message)  # Print full traceback
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
async def root():
    return {"message": "✅ AI Chatbot API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


