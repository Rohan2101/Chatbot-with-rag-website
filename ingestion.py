from operator import index

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore,PineconeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader

embeddings = PineconeEmbeddings(model="multilingual-e5-large")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs","https:/")
        doc.metadata.update({"source":new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents,embeddings,index_name="langchain-doc-index"
    )


if __name__ == "__main__":
    ingest_docs()