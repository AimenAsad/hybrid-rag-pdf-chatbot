import os
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs_from_folder(folder_path):
    pdfs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdfs.append(os.path.join(root, file))

    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        pages = loader.load()
        docs.extend(pages)

    print(f"✅ Loaded {len(docs)} pages from {len(pdfs)} PDFs")
    return docs
