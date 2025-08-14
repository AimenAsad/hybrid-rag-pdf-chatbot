from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_loader import load_pdfs_from_folder

def create_chunks(folder_path):
    docs = load_pdfs_from_folder(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks
