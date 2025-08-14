from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from hybrid_retriever import hybrid_retrieve 

model = ChatOllama(model="llama3:instruct", base_url="http://localhost:11434")
prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

question = "top three most consumed supplements"
docs = hybrid_retrieve(question, k=8)

rag_chain = (
    {
        "context": lambda _: format_docs(docs), 
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke(question)
print("\n🧠 Answer:\n", answer)
