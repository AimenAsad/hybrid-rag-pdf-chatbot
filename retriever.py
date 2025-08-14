from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

PERSIST_DIR = "faiss_index"

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vector_store = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 20, "fetch_k": 100, "lambda_mult": 1}
)
