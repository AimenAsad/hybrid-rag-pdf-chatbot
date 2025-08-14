from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from chunking import create_chunks

all_chunks = None
embeddings = None

def build_vector_store(pdf_folder, persist_dir="faiss_index"):
    global all_chunks, embeddings
    
    all_chunks = create_chunks(pdf_folder)
    print(f"Total chunks created: {len(all_chunks)}")
    for c in all_chunks[:2]:  # preview first 2
        print(c.metadata, c.page_content[:100])

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    single_vector = embeddings.embed_query("test")
    index = faiss.IndexFlatL2(len(single_vector))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(all_chunks)
    vector_store.save_local(persist_dir)

    print(f"✅ Vector store built and saved to '{persist_dir}'")
    return vector_store

if __name__ == "__main__":
    build_vector_store(pdf_folder="RAG-dataset", persist_dir="faiss_index")
