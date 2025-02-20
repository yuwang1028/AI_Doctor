import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ✅ Prevents conflict: Ensure FAISS uses the same embedding model
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

try:
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"❌ ERROR: Failed to load FAISS. Ensure it is correctly built. {str(e)}")
    exit()

# Create retriever from FAISS vectorstore
retriever = VectorStoreRetriever(vectorstore=vectorstore)

# Custom prompt template
system_prompt = (
    "You are an expert in Medical knowledge. "
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# ✅ Prevents conflict: Ensure LLM is properly loaded
def load_llm():
    """Loads OpenAI GPT model for answering queries."""
    try:
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.5,
            max_tokens=512,
            openai_api_key=OPENAI_API_KEY
        )
    except Exception as e:
        print(f"❌ ERROR: Failed to load OpenAI LLM. {str(e)}")
        exit()

# ✅ Prevents conflict: Avoid overwriting FAISS
retrieval_chain = RunnableParallel(
    {"context": retriever, "input": RunnablePassthrough()}
) | prompt | load_llm()

# ✅ Run Safely Without Overwriting FastAPI's FAISS
try:
    user_query = input("Write your query here: ").strip()
    response = retrieval_chain.invoke({"input": user_query})
    print("\n--- Retrieval Q&A ---")
    print("RESULT:", response)
except Exception as e:
    print(f"❌ ERROR: Chatbot failed to respond. {str(e)}")


