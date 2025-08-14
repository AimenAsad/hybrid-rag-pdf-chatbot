import gradio as gr
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from hybrid_retriever import hybrid_retrieve  

model = ChatOllama(model="llama3:instruct", base_url="http://localhost:11434")
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chatbot_interface(user_question):
    if not user_question.strip():
        return "Please enter a question."

    docs = hybrid_retrieve(user_question, k=8)

    rag_chain = (
        {
            "context": lambda _: format_docs(docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain.invoke(user_question)

# Gradio UI
demo = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(placeholder="Ask something from your PDFs...", label="Your Question"),
    outputs=gr.Textbox(label="LLM Answer"),
    title="Chat With PDF",
    description="Hybrid Retrieval (Semantic + Keyword) with FAISS + Ollama."
)

if __name__ == "__main__":
    demo.launch()
