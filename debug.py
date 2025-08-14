# debug_retrieval.py
from vector_embedding import build_vector_store
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

question = "top three most consumed supplements"
docs = vector_store.similarity_search(question, k=8)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:500])
