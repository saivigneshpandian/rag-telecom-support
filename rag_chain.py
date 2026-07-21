"""
Builds the RAG chain:
    merged retriever -> prompt -> Gemini 2.5 Flash -> string output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistanct.
your job is to help customers resolve technical issues with their mobile service.


use only the context below to answer the customer's question.
the context comes from two sources:
- FAQ entries (general policy and hoe to information)
- Past support tickets (real resolved case with step by step resolutions)

if the context does not contain enough information to answer confidently,say mo clearly and suggest the customer call
611 or use the Mytelcom app.

context:{context}
"""


def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
