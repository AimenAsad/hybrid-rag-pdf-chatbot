from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from chunking import create_chunks

# Load chunks from PDFs
all_chunks = create_chunks("RAG-dataset")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

faiss_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50, 'lambda_mult': 0.7}
)

def keyword_search(query, k=5):
    keywords = query.lower().split()
    results = []
    for chunk in all_chunks:
        text = chunk.page_content.lower()
        if any(kw in text for kw in keywords):
            results.append(chunk)
    return results[:k]

def hybrid_retrieve(query, k=5):
    semantic_docs = faiss_retriever.invoke(query)
    keyword_docs = keyword_search(query, k)
    unique_docs = {doc.page_content: doc for doc in (semantic_docs + keyword_docs)}
    return list(unique_docs.values())[:k]
