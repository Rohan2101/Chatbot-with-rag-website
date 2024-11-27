from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain import hub
from langchain_pinecone import PineconeVectorStore,PineconeEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from numpy.f2py.crackfortran import verbose
from typing import List,Dict,Any

from ingestion import embeddings

load_dotenv()
INDEX_NAME = "langchain-doc-index"

def run_llm(query: str,chat_history: List[Dict[str, Any]] = []):
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings)
    chat = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama-3.1-70b-versatile",
        verbose=True,
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat,retrieval_qa_chat_prompt)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,retriever=docsearch.as_retriever(),prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke({"input":query,"chat_history":chat_history})
    new_result = {
        "query":result["input"],
        "result":result["answer"],
        "source_documents": result["context"],
    }
    return new_result

if __name__ == "__main__":
    res = run_llm(query="What is a langchain chain?")
    print(res["answer"])