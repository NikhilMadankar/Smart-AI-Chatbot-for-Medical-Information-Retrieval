import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Import decoupled RAG tool module
from rag_tool import load_vectorstore, medical_knowledge_search

# Step 1: Setup LLM (Groq or HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN):
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=groq_api_key
        )
    elif hf_token:
        return HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            huggingfacehub_api_token=hf_token
        )
    else:
        raise ValueError("Neither GROQ_API_KEY nor HF_TOKEN found in environment. Please set GROQ_API_KEY or HF_TOKEN in your .env file.")

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database via rag_tool helper
db = load_vectorstore()

# Create QA chain (MMR search for diverse, non-redundant retrieval across large documents)
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 10}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("\n" + "="*60)
print("RESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:")
for i, doc in enumerate(response["source_documents"], 1):
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", doc.metadata.get("page_label", "N/A"))
    print(f"\n--- Document {i} ({source}, Page {page}) ---")
    print(doc.page_content.strip())
print("="*60)
