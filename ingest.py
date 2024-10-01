import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Check if vector store already exists
if not os.path.exists("faiss_index"):
    # Load PDFs
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': "cpu"})

    # Create FAISS vector store
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    # Save the vector store to disk
    vector_store.save_local("faiss_index")
    print("Vector store created and saved to disk.")
else:
    print("Vector store already exists. No need to recreate.")


